from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for visualization and plotting
from sklearn.impute import SimpleImputer # Used to fit in missing data
from sklearn import preprocessing # used for label encoder
from sklearn.experimental import enable_iterative_imputer #imputations for NaN values
from sklearn.impute import IterativeImputer #Imputations for NaN values
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('vehicles.csv')

df = df.drop(labels=range(0,27), axis=0)
df.head(30)

### Take outliers out ###
betterdf = df[df['odometer'] < 150000]
betterdf = betterdf[betterdf['odometer'] > 50]
betterdf = betterdf[betterdf['year'] > 1940]
betterdf = betterdf[betterdf['price'] > 500]
betterdf = betterdf[betterdf['price'] < 200000]
print(len(betterdf))
print(len(betterdf))
X = betterdf['year'].values
y = betterdf['price'].values
X = X.astype(float)
X = X.reshape(1,302048)
##########################################


#### Drop Useless Columns ####
onlyNum = {'odometer': betterdf['odometer'], 'year':betterdf['year']}
X = betterdf.drop(columns=['price', 'id', 'url', 'region_url', 'VIN', 'description', 'paint_color', 'image_url', 'county', 'state', 'posting_date']).values
y = betterdf['price']
####################################################################################

### Testing with first 1000 entries ###
X_testing = X[:1000]
y_testing = y[:1000]
#######################################

X_testing = np.transpose(X_testing)

### Note for each feature index ###
#[0] - String - location
#[1] - floats - year ***Dont need a label encoder***
#[2] - String - manufacture
#[3] - String - Model
#[4] - String - Condition
#[5] - String - Engine
#[6] - String - FuelType
#[7] - floats - Odometer ***Dont need a label encoder***
#[8] - String - title status
#[9] - String - Transmission
#[10] - String - Drive
#[11] - String - Size
#[12] - String - Type
#[13] - float - Latitude ***Dont need a label encoder***
#[14] - float - longitude ***Dont need a label encoder***


#### OneHotEncoding ####
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)