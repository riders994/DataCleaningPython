import pandas as pd
from zipfile import ZipFile

zf = ZipFile('data/Train.zip')

df = pd.read_csv('data/Train.csv')

year = df['YearMade']
year = year[year != 1000]

price_v_year = df[['SalePrice', 'YearMade']]
