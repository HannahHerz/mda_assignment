
from pathlib import Path

import pandas as pd

import os
import pandas as pd



app_dir = Path(__file__).parent

df1 = pd.read_csv("https://raw.githubusercontent.com/HannahHerz/mda_assignment/refs/heads/main/MordernDataAnalytics.csv")
df1['ecSignatureDate'] =pd.to_datetime(df1['ecSignatureDate'])

df2 = pd.read_csv("https://raw.githubusercontent.com/HannahHerz/mda_assignment/refs/heads/main/euroSciVoc.csv")
df2['topic'] = df2['euroSciVocPath'].str.extract(r'^/([^/]*)/')
df2_filtered = df2[['projectID', 'topic']]
df2_filtered = df2_filtered.drop_duplicates(subset=['projectID'])
data = df1.merge(df2_filtered, on='projectID', how='left')
data['topic_y'] = data['topic_y'].fillna('not available')
data['topic'] = data['topic_y']
data = data.drop('topic_y', axis=1)

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir("."))