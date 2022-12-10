import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for visualization and plotting
from sklearn import preprocessing # used for label encoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

df = pd.read_csv('vehicles.csv')

### Take outliers out ###
betterdf = df[df['odometer'] < 150000]
betterdf = betterdf[betterdf['odometer'] > 50]
betterdf = betterdf[betterdf['year'] > 1940]
betterdf = betterdf[betterdf['price'] > 500]
betterdf = betterdf[betterdf['price'] < 200000]

y = betterdf['price']
X = betterdf.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor(n_estimators=10, random_state=0)

model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse, mae, r2 = eval_metrics(y_test, pred)

print("RMSE: " + str(rmse))
print("MAE: " + str(mae))
print("r2: " + str(r2))