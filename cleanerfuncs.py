import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation as cv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

glob={}

def run():
    flds = ['SalePrice','ModelID','YearMade','UsageBand','ProductGroup','datasource']
    dumb = ['UsageBand','ProductGroup','ModelID','YearMade']
    model =  train_model(flds,dumb)
    flds2 = ['ModelID','YearMade','UsageBand','ProductGroup','datasource']
    run_testing(model,flds2,dumb)
    pass

def train_model(flsd,dumflds):

    df = pd.read_csv('./Train.csv',index_col='SalesID')
    df_X, df_y = clean_data(df,flsd,dumflds)
    print
    return fit_test(df_X,df_y)

    pass


def clean_data(df,flds,dumflds, test =1):
    ##Creating Initial Test
 #['SalePrice','ModelID','YearMade','MachineHoursCurrentMeter','UsageBand','ProductGroup','datasource']

# [ 'UsageBand', 'ProductGroup']
    # uni = df['MachineID'].unique()
    # for ID in uni:
    #     df['YearMade'][df['MachineID'] == ID] = df['YearMade'][df['MachineID'] == ID][df['YearMade'][df['MachineID'] == ID] > 1000].min()
    df_temp=df.copy()
    df_temp = df_temp[flds]
    #df_temp.dropna(axis=0, inplace = 1)
    simpLabs=[]
    if test:
        simpLabs = df_temp.pop('SalePrice')
    if 'saledate' in flds:
        df_temp['saledate'] = pd.to_datetime(df_temp['saledate'], format = '%m/%d/%Y', exact = 0)

    df_temp = simp_modelID(df_temp)

    for f in dumflds:
        temp_dumb = pd.get_dummies(df_temp[f], prefix = f+'_', drop_first = True, dummy_na=True)

        glob[f]=temp_dumb.columns.values

        df_temp = pd.concat([df_temp, temp_dumb], axis=1)
        df_temp.drop(f, inplace=True, axis=1)
    #df_temp.MachineHoursCurrentMeter[df_temp.MachineHoursCurrentMeter.isnull()] = 0
    print df_temp.shape
    return df_temp, simpLabs

def simp_modelID(df):
    x = df['ModelID'].value_counts()
    x = x[(x.values<500)].index
    df1=df.copy()
    df1['ModelID_use'] = df['ModelID'].astype(str)
    df1['ModelID_use'][df1['ModelID'].isin(x)] = 'Other Val'
    df1['ModelID']= df1['ModelID_use']
    df1.drop('ModelID_use',inplace=True,axis=1)
    print
    return df1

def crossval(X, y):
    model = RandomForestRegressor(n_estimators = 100)
    return cv.cross_val_score(model, X, y)

def fit_test(X,y):

    X_train, X_test, y_train, y_test = cv.train_test_split(X,y)

    model2 = RandomForestRegressor()
    model2.fit(X_train,y_train)
    print model2.score(X_test,y_test)
    return model2

def testDums(trainingcol,testcol,drop = -1, nschem = 'Dummy'):
    if drop == -1:
        uni = trainingcol.unique()[:-1]
    elif drop == 1:
        uni = trainingcol.unique()[1:]
    elif drop == 0:
        uni == trainingcol.unique()
    n = testcol.shape[0]
    res = []
    for dum in uni:
        zs = np.zeros(n)
        zs[testcol == dum] = 1
        res.append(zs)
    res = np.vstack(res)
    colns = [nschem + str(dum) for dum in uni]
    return pd.DataFrame(dict(zip(colns,res)))

def colzip(col1,col2):
    return np.array(["{}_{}".format(a,b) for a,b in zip(col1.astype(str), col2.astype(str))])

def clean_data_test(df,flds,dumflds):
    ##Creating Initial Test
 #['SalePrice','ModelID','YearMade','MachineHoursCurrentMeter','UsageBand','ProductGroup','datasource']

# [ 'UsageBand', 'ProductGroup']

    df_temp=df.copy()
    df_temp = df_temp[flds]
    #df_temp.dropna(axis=0, inplace = 1)

    if 'saledate' in flds:
        df_temp['saledate'] = pd.to_datetime(df_temp['saledate'], format = '%m/%d/%Y', exact = 0)

    df_temp = simp_modelID(df_temp)

    for f in dumflds:
        temp_dumb = pd.get_dummies(df_temp[f], prefix = f+'_', drop_first = True, dummy_na=True)
        temp_dumb = fill_dumbs(temp_dumb,glob.get(f))
        #glob[f]=temp_dumb.columns.values

        df_temp = pd.concat([df_temp, temp_dumb], axis=1)
        df_temp.drop(f, inplace=True, axis=1)
    #df_temp.MachineHoursCurrentMeter[df_temp.MachineHoursCurrentMeter.isnull()] = 0
    print
    return df_temp

def fill_dumbs(temp,temp2):
    for i in temp2:
        if i not in temp.columns.values:
            temp[i]=np.zeros(temp.shape[0])
    print
    return temp

def run_testing(model,flds,dumflds):

    test_df = pd.read_csv('./test.csv', index_col='SalesID')

    test_df = clean_data_test(test_df, flds, dumflds)
    print


    preds = model.predict(test_df)
    IDs = np.array(test_df.index)
    res = pd.DataFrame({
            'SalePrice': preds
        })
    res.index = IDs
    res.index.name = 'SaleID'

    res.to_csv('bad_predictions.csv')
    pass


def TTconsistency(train,test):
    return train[test.columns.values]
