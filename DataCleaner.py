
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn import cross_validation as cv
from sklearn.preprocessing import StandardScaler


class DataCleaner(object):
    """
    Data Cleaning Class object. As of version 1.0, takes in a dictionary of fields, and operations to perform on fields, and applies those operations to a dataframe.
    """
    def __init__(self, fld_info_dict):
        '''
        fld_info_dict: Dictionary of string, string pairs. Each key is the name of the field/column, and the value is the operation to be performed.

        Operations:
        'drop' : Drop the given fields from the DataFrame.
        'label' : Removes label column from frame, and stores separately.
        'dummie' : Turns column into multiple dummy columns to add on to DataFrame
        'boolean' : Modifies column to booleans with specified values. Ex.'gender' : 'boolean={male:1,female:0}' would change a column named 'gender' in a DataFrame to a column that contained 1s and 0s for the males and females respectively.
        '''

        self.fld_info_dict = fld_info_dict
        self.y_val_true = False
        self.dummy_dict = {}
        self.std_scaler = None
        self.y = None


    def clean(self, df):
        '''
        Takes a Pandas DataFrame and does the modifications specified in the Initalize
        Returns a NP matrix of X vlaues
        if label field exists returns NP matrix of X and the y values
        X,y format.
        '''
        print self.fld_info_dict
        for fld, funct in self.fld_info_dict.iteritems():
            if funct == 'drop':  #drop field
                df.drop(fld, inplace=True, axis=1)

            elif funct == 'label':
                if fld in df.columns.values:
                    self.y = df.pop(fld)
                    self.y_val_true =  True

            elif funct == 'dummie':
                '''Makes the dummies for the given fields.  Saves the fields used in a dummy dictionary to compare when testing files are cleaned. '''
                dumbs = pd.get_dummies(df[fld], prefix = fld+'_', dummy_na=True)
                if self.dummy_dict.get(fld) == None:
                    self.dummy_dict[fld]=dumbs.columns.values
                else:
                    dumbs = _test_check_dumbs(dumbs,self.dummy_dict.get(f))

                df = pd.concat([df, dumbs], axis=1)
                df.drop(fld, inplace=True, axis=1)

            elif funct.split('=')[0] == 'boolean':
                blvl = funct.split('=')[1]
                items = blvl.strip('{}').split(',')
                pairs = [item.split(':',1) for item in items]
                boolean_dict = dict((k,eval(v)) for (k,v) in pairs)
                df[fld] = df[fld].map(boolean_dict)

        return df, self.y


    def _test_check_dumbs(self, temp,temp2):
        for i in temp2:
            if i not in temp.columns.values:
                temp[i]=np.zeros(temp.shape[0])
        for i in temp.columns.values:
            if i not in temp2:
                temp.drop(i, axis = 1, inplace = True)
        return temp


    def normalize_data(self, X):
        '''
        call and pass in data to normalize X data.
        keeps track of the StandardScaler so you dont need to
        '''
        if self.std_scaler == None:
            self.std_scaler = StandardScaler().fit(X)
        return self.std_scaler.transform(X)
