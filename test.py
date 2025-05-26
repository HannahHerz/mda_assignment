import pandas as pd, numpy as np; from sklearn.model_selection import train_test_split; from sklearn.compose import ColumnTransformer, TransformedTargetRegressor; from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, PowerTransformer, OrdinalEncoder; from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS; from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier; from sklearn.decomposition import LatentDirichletAllocation; from sklearn.pipeline import Pipeline, FeatureUnion; from sklearn.impute import SimpleImputer; from sklearn.base import BaseEstimator, TransformerMixin; from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error; from sklearn.svm import SVC; from xgboost import XGBRegressor; import matplotlib.pyplot as plt; import openpyxl; from joblib import parallel_backend; import pickle

df = pd.read_excel("MordernDataAnalytics.xlsx"); geo_groups = {'Western_Europe':{'DE','FR','BE','NL','LU','CH','AT','LI'},'Northern_Europe':{'UK','IE','SE','FI','DK','IS','NO','EE','LV','LT'},'Southern_Europe':{'IT','ES','PT','EL','MT','CY','SI'},'Eastern_Europe':{'PL','CZ','SK','HU','RO','BG','RS','UA','AL','MK','ME','XK','HR','MD','GE','BA'},'Africa':{...},'Asia':{...},'Oceania':{'AU','NZ','FJ','MH','PG','NC'},'Americas':{'US','CA','BR','AR','CO','CL','MX','PE','UY','BO','CR','PA','GT','SV','PY','EC','VE','DO','HT','SR','AW','BQ','AI','GU'}}
for r,c in geo_groups.items(): df[f'{r}_count']=df['country'].str.split(';').apply(lambda L:sum(x.strip() in c for x in L))
df['num_countries']=df['country'].str.count(';')+1; df['num_organisations']=df['organisationID'].str.count(';')+1
df['num_sme']=df['SME'].str.split(';').apply(lambda L:sum(e.lower()=='true' for e in L if e.strip()))
df['role_list']=df['role'].str.split(';').apply(lambda L:[r.strip() for r in L]); df['n_participant']=df['role_list'].apply(lambda L:L.count('participant')); df['n_associatedPartner']=df['role_list'].apply(lambda L:L.count('associatedPartner')); df['n_thirdParty']=df['role_list'].apply(lambda L:L.count('thirdParty'))
for col in ['startDate','endDate','ecSignatureDate']: df[col]=pd.to_datetime(df[col],errors='coerce')
df['duration_days']=(df.endDate-df.startDate).dt.days; df['start_year']=df.startDate.dt.year; df['start_month']=df.startDate.dt.month
df2=pd.read_excel("euroSciVoc.xlsx",usecols=['projectID','euroSciVocPath']); df2['euroSciVoxTopic']=df2['euroSciVocPath'].str.extract(r'^/([^/]+)/?'); df=pd.merge(df,df2[['projectID','euroSciVoxTopic']].drop_duplicates(),on='projectID',how='left'); df['euroSciVoxTopic'].fillna('not available',inplace=True)
df.to_csv("df_eu.csv",index=False)
target=df['ecMaxContribution']; X_train,X_test,y_train,y_test=train_test_split(df,target,test_size=0.2,random_state=42)

# compute org past means
train_long=(X_train[['projectID','organisationID','ecContribution_list','startDate']].assign(clean_orgs=lambda d:d.organisationID.astype(str).str.strip('[]').str.replace(r'[;,]',';'),
 clean_ecs=lambda d:d.ecContribution_list.astype(str).str.strip('[]').str.replace(r'[;,]',';'))
 .assign(org_list=lambda d:d.clean_orgs.str.split(';').apply(lambda L:[o.strip()for o in L if o.strip()]),
         ec_list=lambda d:d.clean_ecs.str.split(';').apply(lambda L:[float(x)for x in L if x.strip()]))
 .explode(['org_list','ec_list']).rename(columns={'ec_list':'ecContribution','org_list':'organisationID'}).sort_values(['organisationID','startDate']))
train_long['cum_sum']=train_long.groupby('organisationID')['ecContribution'].cumsum().shift(1).fillna(0); train_long['cum_count']=train_long.groupby('organisationID')['ecContribution'].cumcount()
train_long['past_mean']=(train_long.cum_sum/train_long.cum_count.replace(0,np.nan)).fillna(0)
org_dim=train_long.groupby('organisationID')['past_mean'].last().reset_index().rename(columns={'past_mean':'org_past_mean_ec'})

# stop words and encoders
extra_stops={'project','new','study','research','based','use'}; stops=list(ENGLISH_STOP_WORDS.union(extra_stops))
class SupervisedLDATopicEncoder(BaseEstimator,TransformerMixin):
    def __init__(s,n_components=10,vectorizer_params=None,lda_params=None): s.n_components,n_params,l_params=n_components,vectorizer_params or {},lda_params or {}
    def fit(s,X,y):
        dtm=CountVectorizer(**s.vectorizer_params).fit_transform(X.objective)
        lda=LatentDirichletAllocation(n_components=s.n_components,**s.lda_params).fit(dtm)
        doc_topic=lda.transform(dtm); y_arr=np.array(y).reshape(-1,1)
        s.topic_means_=(doc_topic*y_arr).sum(0)/doc_topic.sum(0); s.vectorizer_,s.lda_=CountVectorizer(**s.vectorizer_params),lda
        return s
    def transform(s,X): return (s.lda_.transform(s.vectorizer_.transform(X.objective))*s.topic_means_).sum(1).reshape(-1,1)

class OrgAvgPastEC(BaseEstimator,TransformerMixin):
    def __init__(s,org_dim): s.org_dim=(org_dim.set_index('organisationID')['org_past_mean_ec'].to_dict()if hasattr(org_dim,'set_index')else org_dim)
    def fit(s,X,y=None):return s
    def transform(s,X):
        m=X.apply(lambda r:np.mean([s.org_dim.get(o.strip(),0)for o in str(r.organisationID).strip('[]').replace(',',';').split(';') if o.strip()]),axis=1)
        return m.fillna(0).values.reshape(-1,1)

numeric_cols=['duration_days','n_participant','n_associatedPartner','n_thirdParty']+[f"{r}_count" for r in ['Western_Europe','Eastern_Europe','Northern_Europe','Southern_Europe','Africa','Asia','Oceania','Americas']]+['num_countries','num_organisations','num_sme','start_year']
preprocessor=FeatureUnion([
    ('num',ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),numeric_cols)])),
    ('cat',ColumnTransformer([('c',Pipeline([('i',SimpleImputer(fill_value='__MISSING__')),('e',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),['fundingScheme','masterCall','euroSciVoxTopic'])])),
    ('month',Pipeline([('c', FunctionTransformer(lambda X: np.vstack([np.sin(2*np.pi*X.start_month/12),np.cos(2*np.pi*X.start_month/12)]).T)),('s',StandardScaler())])),
    ('topic',SupervisedLDATopicEncoder(10,{'stop_words':stops,'min_df':5,'ngram_range':(1,2)},{'max_iter':20,'learning_method':'online','random_state':42,'n_jobs':-1})),
    ('orgavg',Pipeline([('o',OrgAvgPastEC(org_dim)),('s',StandardScaler())]))
],n_jobs=1)

# train/test features
with parallel_backend('threading'):
    Xtr=preprocessor.fit_transform(X_train,y_train); Xte=preprocessor.transform(X_test)
ywl=FunctionTransformer(np.log1p,inverse_func=np.expm1).fit_transform(y_train).ravel(); thr=13
is_large=(ywl>thr).astype(int)
clf=RandomForestClassifier(100,42).fit(Xtr,is_large)
is_large_test=(FunctionTransformer(np.log1p).transform(y_test).ravel()>thr).astype(int)
print("RF acc:",accuracy_score(is_large_test,clf.predict(Xte)))
svm=SVC(random_state=42).fit(Xtr,is_large)
print("SVM acc:",accuracy_score(is_large_test,svm.predict(Xte)))

# regressors
small=TransformedTargetRegressor(RandomForestRegressor(200,42,-1),FunctionTransformer(np.log1p,np.expm1))
large=RandomForestRegressor(200,42,-1)
mask=is_large==0; small.fit(Xtr[mask],y_train[mask]); large.fit(Xtr[~mask],y_train[~mask])
y_pred=np.empty(len(y_test)); mask_small=(is_large_test==0); y_pred[mask_small]=small.predict(Xte[mask_small]); y_pred[~mask_small]=large.predict(Xte[~mask_small])
print("R2",r2_score(y_test,y_pred),"MAE",mean_absolute_error(y_test,y_pred))

# save models
for name,mdl in [('small',small),('large',large),('clf',clf)]: pickle.dump(mdl,open(f"{name}_model.pkl",'wb'))
class HierarchicalGrantModel:
    def __init__(s,c,sm,l): s.c,s.sm,s.l=c,sm,l
    def predict(s,X):
        m=s.c.predict(X); out=np.empty(len(X))
        if any(~m): out[~m]=s.sm.predict(X[~m])
        if any(m): out[m]=s.l.predict(X[m])
        return out
pickle.dump(HierarchicalGrantModel(clf,small,large),open('hierarchical_grant_model.pkl','wb'))
