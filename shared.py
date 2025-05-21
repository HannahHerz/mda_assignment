
from pathlib import Path

import pandas as pd
import openpyxl

app_dir = Path(__file__).parent
tips = pd.read_csv(app_dir / "tips.csv")
data = pd.read_excel("MordernDataAnalytics.xlsx")