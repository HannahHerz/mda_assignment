# model.py

import joblib
import pandas as pd
import numpy as np
import sys
import os
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer classes (must be defined for pickle loading)
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
        # Avoid division by zero
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
        # Classify small vs large
        is_large_pred = self.classifier.predict(X)
        
        # Initialize predictions array
        y_pred = np.empty(len(X), dtype=float)
        
        # Get masks
        mask_small = (is_large_pred == 0)
        mask_large = (is_large_pred == 1)
        
        # Make predictions
        if np.any(mask_small):
            y_pred[mask_small] = self.small_model.predict(X[mask_small])
        if np.any(mask_large):
            y_pred[mask_large] = self.large_model.predict(X[mask_large])
            
        return y_pred

# Fix for pickle loading - make classes available in __main__ namespace
if __name__ != '__main__':
    # When importing as a module, add classes to main namespace for pickle compatibility
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

# Function to safely load models with error handling
def load_model_safely(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Make sure all pickle files are in the correct directory.")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load the trained models
preprocessor = load_model_safely("preprocessor.pkl")
size_clf     = load_model_safely("classifier_model.pkl")
small_model  = load_model_safely("small_grant_model.pkl")
large_model  = load_model_safely("large_grant_model.pkl")

# Check if all models loaded successfully
models_loaded = all([preprocessor, size_clf, small_model, large_model])

def make_input_df(inputs: dict) -> pd.DataFrame:
    """
    Create a DataFrame from input dictionary that matches the training data format
    """
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

    # Check if country counts are already provided in inputs
    country_counts_provided = any(f"{region}_count" in inputs for region in geo_groups.keys())
    
    if country_counts_provided:
        # Use provided country counts
        for region in geo_groups.keys():
            row[f"{region}_count"] = int(inputs.get(f"{region}_count", 0))
        row['num_countries'] = int(inputs.get('num_countries', 0))
    else:
        # Calculate country counts from countries string (fallback)
        codes = [c.strip().upper() for c in inputs.get('countries','').split(';') if c.strip()]
        for region, countries in geo_groups.items():
            row[f"{region}_count"] = sum(code in countries for code in codes)
        row['num_countries'] = len(codes)

    return pd.DataFrame([row])

def predict_funding(inputs: dict) -> float:
    """
    Make a funding prediction based on input parameters
    """
    if not models_loaded:
        raise Exception("Models not loaded properly. Check that all .pkl files are in the correct directory.")
    
    try:
        # Create input DataFrame
        df = make_input_df(inputs)
        
        # Transform features using the preprocessor
        Xf = preprocessor.transform(df)
        
        # Classify as small or large grant
        is_large = bool(size_clf.predict(Xf)[0])
        
        # Select appropriate model and make prediction
        model = large_model if is_large else small_model
        prediction = float(model.predict(Xf)[0])
        
        # Ensure prediction is positive
        return max(0, prediction)
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

# For debugging purposes
def debug_prediction(inputs: dict):
    """
    Debug function to see intermediate steps of prediction
    """
    try:
        print("Input data:")
        df = make_input_df(inputs)
        print(df.to_string())
        
        print("\nTransformed features shape:")
        Xf = preprocessor.transform(df)
        print(Xf.shape)
        
        print("\nClassification (is_large):")
        is_large = bool(size_clf.predict(Xf)[0])
        print(is_large)
        
        print("\nFinal prediction:")
        model = large_model if is_large else small_model
        prediction = float(model.predict(Xf)[0])
        print(f"€{prediction:,.2f}")
        
        return prediction
        
    except Exception as e:
        print(f"Debug failed: {e}")
        return None