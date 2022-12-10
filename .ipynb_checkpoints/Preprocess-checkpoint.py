import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('vehicles.csv')
df = df.drop(labels=range(0,27), axis=0)

### Take outliers out ###
betterdf = df[df['odometer'] < 150000]
betterdf = betterdf[betterdf['odometer'] > 50]
betterdf = betterdf[betterdf['year'] > 1940]
betterdf = betterdf[betterdf['price'] > 500]
betterdf = betterdf[betterdf['price'] < 200000]

#### Drop Useless Columns ####
betterdf = betterdf.drop(['id', 'url', 'region_url', 'VIN', 'description', 'paint_color', 'image_url', 'county', 'state', 'posting_date'], axis=1)
betterdf.info()

columns = [col for col in betterdf.columns if betterdf[col].dtype=="O"]

betterdf_cat = betterdf.loc[:, betterdf.dtypes==object]
betterdf_num = betterdf.loc[:, betterdf.dtypes==float]

betterdf_cat.fillna("unknown")
betterdf_num.fillna(0.0)

for column in columns:
    tempdf = pd.get_dummies(betterdf_cat[column], prefix=column)

    betterdf_cat = pd.merge(
        left=betterdf_cat,
        right=tempdf,
        left_index=True,
        right_index=True,
    )

    betterdf_cat = betterdf_cat.drop(columns=column)

betterdf = pd.concat([betterdf_num, betterdf_cat], axis=1, join='inner')

betterdf.info()

betterdf.to_csv('vehicles_cleaned.csv')