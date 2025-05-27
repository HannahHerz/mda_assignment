import joblib
import pandas as pd
import numpy as np
import os
import pickle
import requests
from io import BytesIO
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer classes (must be defined for pickle loading)
class OrgAvgPastEC(BaseEstimator, TransformerMixin):
    def __init__(self, org_dim):
        if hasattr(org_dim, 'set_index'):
            org_dim = org_dim.set_index('organisationID')['org_past_mean_ec'].to_dict()
        self.org_dim = org_dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        proj_to_means = {}
        for _, row in X.iterrows():
            pid = row['projectID']
            orgs = (
                str(row['organisationID'])
                   .strip('[]')
                   .replace(r',', ';')
                   .split(';')
            )
            orgs = [o.strip() for o in orgs if o.strip()]
            means = [ self.org_dim.get(o, 0.0) for o in orgs ]
            proj_to_means[pid] = float(np.mean(means)) if means else 0.0

        arr = X['projectID'].map(proj_to_means).fillna(0.0).values
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
        self.vectorizer_ = CountVectorizer(
            stop_words=self.stop_words,
            **self.vectorizer_params
        )
        dtm = self.vectorizer_.fit_transform(texts)

        self.lda_ = LatentDirichletAllocation(
            n_components=self.n_components,
            **self.lda_params
        ).fit(dtm)

        doc_topic = self.lda_.transform(dtm)
        y_arr     = np.array(y).reshape(-1,1)
        sums      = (doc_topic * y_arr).sum(axis=0)
        weights   = doc_topic.sum(axis=0)
        weights = np.where(weights == 0, 1e-8, weights)
        self.topic_means_ = sums / weights
        return self

    def transform(self, X):
        dtm       = self.vectorizer_.transform(X['objective'])
        doc_topic = self.lda_.transform(dtm)
        return (doc_topic * self.topic_means_).sum(axis=1).reshape(-1,1)

class HierarchicalGrantModel:
    def __init__(self, classifier, small_model, large_model):
        self.classifier = classifier
        self.small_model = small_model
        self.large_model = large_model
    
    def predict(self, X):
        is_large_pred = self.classifier.predict(X)
        y_pred = np.empty(len(X), dtype=float)
        mask_small = (is_large_pred == 0)
        mask_large = (is_large_pred == 1)
        if np.any(mask_small):
            y_pred[mask_small] = self.small_model.predict(X[mask_small])
        if np.any(mask_large):
            y_pred[mask_large] = self.large_model.predict(X[mask_large])
        return y_pred

# Expose classes for pickle compatibility
import __main__
__main__.OrgAvgPastEC = OrgAvgPastEC
__main__.CyclicalEncoder = CyclicalEncoder
__main__.SupervisedLDATopicEncoder = SupervisedLDATopicEncoder
__main__.HierarchicalGrantModel = HierarchicalGrantModel

# Country group definitions
geo_groups = {
    'Western_Europe': {'DE','FR','BE','NL','LU','CH','AT','LI'},
    'Northern_Europe': {'UK','IE','SE','FI','DK','IS','NO','EE','LV','LT'},
    'Southern_Europe': {'IT','ES','PT','EL','MT','CY','SI'},
    'Eastern_Europe': {'PL','CZ','SK','HU','RO','BG','RS','UA','AL','MK','ME','XK','HR','MD','GE','BA'},
    'Africa': {'ZA','KE','UG','TN','GH','MA','TZ','EG','SN','CD','MZ','RW','BF','ZM','CI','CM','ET','NG','DZ','AO','GN','BJ','GA','MW','ML','BI','MU','ST','LR','ZW','CG','GW','NE','LY','GQ','SD','LS','TD','DJ'},
    'Asia': {'IL','TR','IN','CN','JP','KR','TH','SG','LB','TW','UZ','AM','VN','MY','KZ','PK','AZ','HK','ID','JO','BD','KG','IR','PS','MN','KH','TJ','IQ','TM','NP','KW','QA','AF','BT','MO','MV','LA','LK'},
    'Oceania': {'AU','NZ','FJ','MH','PG','NC'},
    'Americas': {'US','CA','BR','AR','CO','CL','MX','PE','UY','BO','CR','PA','GT','SV','PY','EC','VE','DO','HT','SR','AW','BQ','AI','GU'},
}

# Safe model loading without prints

def load_model_safely(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        return None
    except Exception:
        return None

# GitHub LFS loader without prints
def load_from_github_lfs(filename, base_url, fallback_local=True):
    url = base_url + filename
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise Exception
        content_text = response.text
        if content_text.startswith('version https://git-lfs.github.com/spec/v1'):
            lines = content_text.strip().split('\n')
            oid = None
            for line in lines:
                if line.startswith('oid sha256:'):
                    oid = line.split(':', 1)[1]
            if oid:
                lfs_url = f"https://media.githubusercontent.com/media/HannahHerz/mda_assignment/main/{filename}"
                lfs_response = requests.get(lfs_url, timeout=30)
                if lfs_response.status_code == 200:
                    return pickle.load(BytesIO(lfs_response.content))
        else:
            return pickle.load(BytesIO(response.content))
    except Exception:
        if fallback_local:
            return load_model_safely(filename)
        return None

# Initialize models
preprocessor = None
size_clf = None
small_model = None
large_model = None
models_loaded = False

# Model validation helper
def validate_model(model):
    if model is None:
        return False
    try:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(model)
        return True
    except Exception:
        return False

# Load all models

def load_models():
    global preprocessor, size_clf, small_model, large_model, models_loaded
    base_url = "https://raw.githubusercontent.com/HannahHerz/mda_assignment/refs/heads/main/"
    preprocessor = load_model_safely("preprocessor.pkl") or load_from_github_lfs("preprocessor.pkl", base_url, False)
    size_clf     = load_model_safely("classifier_model.pkl") or load_from_github_lfs("classifier_model.pkl", base_url, False)
    small_model  = load_model_safely("small_grant_model.pkl") or load_from_github_lfs("small_grant_model.pkl", base_url, False)
    large_model  = load_model_safely("large_grant_model.pkl") or load_from_github_lfs("large_grant_model.pkl", base_url, False)
    models_loaded = all([validate_model(preprocessor), validate_model(size_clf), validate_model(small_model), validate_model(large_model)])

# Dataframe creation

def make_input_df(inputs: dict) -> pd.DataFrame:
    start = pd.to_datetime(inputs['start_date'], dayfirst=True)
    dur   = int(inputs['duration_days'])
    row = {
        'projectID':           inputs.get('projectID', 'PRED_001'),
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
        'masterCall':          inputs.get('masterCall', '__MISSING__'),
        'euroSciVoxTopic':     inputs.get('euroSciVoxTopic','not available'),
        'objective':           inputs.get('objective', ''),
        'organisationID':      inputs.get('organisationID', ''),
    }
    codes = [c.strip().upper() for c in inputs.get('countries','').split(';') if c.strip()]
    for region, countries in geo_groups.items():
        row[f"{region}_count"] = sum(code in countries for code in codes)
    row['num_countries'] = len(codes)
    return pd.DataFrame([row])

# Prediction function

def predict_funding(inputs: dict) -> float:
    global models_loaded
    if not models_loaded:
        load_models()
    if not models_loaded:
        raise Exception("Models not loaded properly.")
    df = make_input_df(inputs)
    Xf = preprocessor.transform(df)
    is_large = bool(size_clf.predict(Xf)[0])
    model = large_model if is_large else small_model
    prediction = float(model.predict(Xf)[0])
    return max(0, prediction)

# Debugging (no prints)
def debug_prediction(inputs: dict):
    global models_loaded
    if not models_loaded:
        load_models()
    df = make_input_df(inputs)
    Xf = preprocessor.transform(df)
    is_large = bool(size_clf.predict(Xf)[0])
    model = large_model if is_large else small_model
    return float(model.predict(Xf)[0])

if __name__ == "__main__":
    load_models()
