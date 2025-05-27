import pandas as pd
import openpyxl

read_file1 = pd.read_excel ("MordernDataAnalytics.xlsx")

# Write the dataframe object
# into csv file
read_file1.to_csv ("MordernDataAnalytics.csv", 
                  index = None,
                  header=True)
  

read_file2 = pd.read_excel ("euroSciVoc.xlsx")

# Write the dataframe object
# into csv file
read_file2.to_csv ("euroSciVoc.csv", 
                  index = None,
                  header=True)
  