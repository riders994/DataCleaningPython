import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation as cv
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./Train.csv', index_col='SalesID')

##Creating Initial Test
simpledf = df[['SalePrice','ModelID','YearMade','MachineHoursCurrentMeter','UsageBand','saledate','state','fiModelDesc','fiBaseModel','fiModelSeries','ProductGroup','datasource']]
simpledf.dropna(axis=0, inplace = 1)
simpLabs = simpledf['SalePrice']
simpledf['saledate'] = pd.to_datetime(simpledf['saledate'], format = '%m/%d/%Y', exact = 0)


UsageDums = pd.get_dummies(simpledf['UsageBand'], prefix='Usage', drop_first=1)
ProdDums = pd.get_dummies(simpledf['ProductGroup'], prefix='Prod_', drop_first=1)
StateDums = pd.get_dummies(simpledf['state'], drop_first=1)
SourceDums = pd.get_dummies(simpledf['datasource'], drop_first=1)

SourceDums = pd.get_dummies(simpledf['datasource'], drop_first=1)

DumDf = simpledf.drop(['fiModelSeries','ModelID','UsageBand','state','ProductGroup','fiModelDesc','fiBaseModel','saledate','SalePrice'], axis=1)

DumDf = pd.concat([DumDf, UsageDums], axis=1)
DumDf = pd.concat([DumDf, ProdDums], axis=1)
#DumDf = pd.concat([DumDf, StateDums], axis=1)
DumDf = pd.concat([DumDf, SourceDums], axis=1)

def crossval():
    model = LinearRegression()
    return cv.cross_val_score(model,DumDf, simpLabs)

X_train, X_test, y_train, y_test = cv.train_test_split(DumDf,simpLabs)

model = LinearRegression()
model.fit(X_train,y_train)

model.score(X_test, y_test)

def testDums(trainingcol,testcol,drop = -1, nschem = 'Dummy'):
    if drop == -1:
        uni = trainingcol.unique()[:-1]
    elif drop == 1:
        uni = trainingcol.unique()[1:]
    elif drop == 0:
        uni = = trainingcol.unique()
    n = testcol.shape[0]
    res = [np.zeros(n)[testcol == dum] = 1 for dum in uni]
    res = np.vstack(res)
    colns = [nschem + str(dum) for dum in uni]
    return res, colns

def colzip(col1,col2):
    return np.array(["{}_{}".format(a,b) for a,b in zip(col1.astype(str), col2.astype(str))])

testdf = pd.read_csv('./test.csv', index_col='SalesID')

print 'boop'
simpletest = testdf[['ModelID','YearMade','MachineHoursCurrentMeter','UsageBand','saledate','state','fiModelDesc','fiBaseModel','ProductGroup','datasource']]

simpletest = simpletest.dropna(axis=0)

tUsageDums = pd.get_dummies(simpletest['UsageBand'], prefix='Usage', drop_first=1)
tProdDums = pd.get_dummies(simpletest['ProductGroup'], prefix='Prod_', drop_first=1)
tStateDums = pd.get_dummies(simpletest['state'], drop_first=1)
tSourceDums = pd.get_dummies(simpletest['datasource'], drop_first=1)

testDums = simpletest.drop(['ModelID','UsageBand','state','ProductGroup','fiModelDesc','fiBaseModel','saledate'], axis=1)

testDums = pd.concat([testDums, tUsageDums], axis=1)
testDums = pd.concat([testDums, tProdDums], axis=1)
#testDums = pd.concat([testDums, tStateDums], axis=1)
testDums = pd.concat([testDums, tSourceDums], axis=1)
testDums['136'] = np.zeros(4031)

preds = model.predict(testDums)
IDs = np.array(testDums.index)
res = pd.DataFrame({
        'SalePrice': preds
    })
res.index = IDs
res.index.name = 'SaleID'

res.to_csv('bad_predictions.csv')
