# model.py

import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 0) redefine all classes for pickles (used at inference time)
class OrgAvgPastEC(BaseEstimator, TransformerMixin):
    def __init__(self, org_dim):
        self.org_dim = (
            org_dim
            if isinstance(org_dim, dict)
            else org_dim.set_index('organisationID')['org_past_mean_ec'].to_dict()
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def mean_for(cell):
            orgs = [o.strip() for o in str(cell).split(';') if o.strip()]
            vals = [self.org_dim.get(o, 0.0) for o in orgs]
            return float(np.mean(vals)) if vals else 0.0

        arr = X['organisationID'].apply(mean_for).values
        return arr.reshape(-1,1)

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col, period):
        self.col = col
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vals = X[self.col].astype(float).values
        sin = np.sin(2 * np.pi * vals / self.period)
        cos = np.cos(2 * np.pi * vals / self.period)
        return np.vstack([sin, cos]).T

class SupervisedLDATopicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, vectorizer_params=None, lda_params=None):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        extra = {'project','new','study','research','based','use'}
        self.stop_words = list(ENGLISH_STOP_WORDS.union(extra))
        self.n_components      = n_components
        self.vectorizer_params = vectorizer_params or {}
        self.lda_params        = lda_params or {}

    def fit(self, X, y):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        texts = X['objective']
        self.vectorizer_ = CountVectorizer(**self.vectorizer_params)
        dtm = self.vectorizer_.fit_transform(texts)

        self.lda_ = LatentDirichletAllocation(
            n_components=self.n_components,
            **self.lda_params
        ).fit(dtm)

        doc_topic = self.lda_.transform(dtm)
        y_arr     = np.array(y).reshape(-1,1)
        sums      = (doc_topic * y_arr).sum(axis=0)
        weights   = doc_topic.sum(axis=0)
        self.topic_means_ = sums / weights
        return self

    def transform(self, X):
        dtm       = self.vectorizer_.transform(X['objective'])
        doc_topic = self.lda_.transform(dtm)
        return (doc_topic * self.topic_means_).sum(axis=1).reshape(-1,1) 

# 1) country groups
geo_groups = {
    'Western Europe': {'DE','FR','BE','NL','LU','CH','AT','LI'},
    'Northern Europe': {'UK','IE','SE','FI','DK','IS','NO','EE','LV','LT'},
    'Southern Europe': {'IT','ES','PT','EL','MT','CY','SI'},
    'Eastern Europe': {'PL','CZ','SK','HU','RO','BG','RS','UA','AL','MK','ME','XK','HR','MD','GE','BA'},
    'Africa':          {'ZA','KE','UG','TN','GH','MA','TZ','EG','SN','CD','MZ','RW','BF','ZM','CI','CM','ET','NG','DZ','AO','GN','BJ','GA','MW','ML','BI','MU','ST','LR','ZW','CG','GW','NE','LY','GQ','SD','LS','TD','DJ'},
    'Asia':            {'IL','TR','IN','CN','JP','KR','TH','SG','LB','TW','UZ','AM','VN','MY','KZ','PK','AZ','HK','ID','JO','BD','KG','IR','PS','MN','KH','TJ','IQ','TM','NP','KW','QA','AF','BT','MO','MV','LA','LK'},
    'Oceania':         {'AU','NZ','FJ','MH','PG','NC'},
    'Americas':        {'US','CA','BR','AR','CO','CL','MX','PE','UY','BO','CR','PA','GT','SV','PY','EC','VE','DO','HT','SR','AW','BQ','AI','GU'},
}

# 2) load retrained pipeline and models
preprocessor = joblib.load("mda_assignment/models/preprocessor.pkl")
size_clf     = joblib.load("mda_assignment/models/size_classifier.pkl")
small_model  = joblib.load("mda_assignment/models/small_grant_model.pkl")
large_model  = joblib.load("mda_assignment/models/large_grant_model.pkl")

# 3) input df
def make_input_df(inputs: dict) -> pd.DataFrame:
    start = pd.to_datetime(inputs['start_date'], dayfirst=True)
    dur   = int(inputs['duration_days'])
    row = {
        'startDate':           start,
        'endDate':             start + pd.Timedelta(days=dur),
        'duration_days':       dur,
        'start_year':          start.year,
        'start_month':         start.month,
        'n_participant':       int(inputs.get('n_participant', 0)),
        'n_associatedPartner': int(inputs.get('n_associatedPartner', 0)),
        'n_thirdParty':        int(inputs.get('n_thirdParty', 0)),
        'num_organisations':   int(inputs.get('num_organisations', 0)),
        'num_sme':             int(inputs.get('num_sme', 0)),
        'fundingScheme':       inputs.get('fundingScheme', '__MISSING__'),
        'masterCall':          inputs.get('masterCall',    '__MISSING__'),
        'euroSciVoxTopic':     inputs.get('euroSciVoxTopic','not available'),
        'objective':           inputs.get('objective',      ''),
        'organisationID':      inputs.get('organisationID', ''),
    }

    codes = [c.strip().upper() for c in inputs.get('countries','').split(';') if c.strip()]
    for region, countries in geo_groups.items():
        row[f"{region}_count"] = sum(code in countries for code in codes)
    row['num_countries'] = len(codes)

    return pd.DataFrame([row])

# 4) predict funding
def predict_funding(inputs: dict) -> float:
    df = make_input_df(inputs)
    Xf = preprocessor.transform(df)
    is_large = bool(size_clf.predict(Xf)[0])
    model    = large_model if is_large else small_model
    return float(model.predict(Xf)[0])

# 5) check
if __name__ == "__main__":
    sample = {
        'start_date':            '01/02/2025',
        'duration_days':         1825,
        'n_participant':         8,
        'n_associatedPartner':   2,
        'n_thirdParty':          0,
        'num_organisations':     8,
        'num_sme':               1,
        'fundingScheme':         'HORIZON-JU-RIA',
        'masterCall':            'HORIZON-JU-GH-EDCTP3-2023-02-two-stage',
        'euroSciVoxTopic':       'medical and health sciences',
        'objective':             'Develop the first POC serological RDT for P. vivax.',
        'countries':             'MG;SN;ET;UK;CH;FR;UK;AU',
        'organisationID':        '986872084;999542806;889740358;999912667;999679964;999993080;999668809;954560414',
    }
    print(f"Predicted €: {predict_funding(sample):,.0f}")
