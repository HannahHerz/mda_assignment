
from pathlib import Path

import pandas as pd
import openpyxl

app_dir = Path(__file__).parent

df1 = pd.read_excel("mda_assignment/MordernDataAnalytics.xlsx")
df1['ecSignatureDate'] =pd.to_datetime(df1['ecSignatureDate'])

df2 = pd.read_excel("mda_assignment/euroSciVoc.xlsx")
df2['topic'] = df2['euroSciVocPath'].str.extract(r'^/([^/]*)/')
df2_filtered = df2[['projectID', 'topic']]
df2_filtered = df2_filtered.drop_duplicates(subset=['projectID'])
data = df1.merge(df2_filtered, on='projectID', how='left')
data['topic_y'] = data['topic_y'].fillna('not available')
data['topic'] = data['topic_y']
data = data.drop('topic_y', axis=1)

