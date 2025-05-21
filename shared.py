
from pathlib import Path

import pandas as pd
import openpyxl

app_dir = Path(__file__).parent

data = pd.read_excel("MordernDataAnalytics.xlsx")
data['ecSignatureDate'] =pd.to_datetime(data['ecSignatureDate'])
