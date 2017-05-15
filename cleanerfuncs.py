#Current Version: 1.0

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv




def clean_Data(df, fields = False, dummys = False, threshold = 0, y = False, dates = False, linear = False, dateform = '%m/%d/%Y', **kwargs):
    """
    Data cleaning function. As of version 1.1, will take fields from original
    data frame, make dummies apropriately, and set dates, and can convert string
    columns to numeric, stripping currency and percents.

    Requirements:

    df: Pandas dataframe to be cleaned. Will add in Numpy functionality at later
        date.

    Possible Variables:

    fields: List of fields (strings) from original dataframe to be left as is.
            If left blank, will work on dummies only, or will return original
            dataframe.

    dummys: List of fields (strings) from original dataframe to be turned into
            dummy variables. If left blank, will return as specified by fields
            and dates.

    threshold: Minimum threshold for dummy variables. If there are fewer
               indicators than threshold, will change to 'Unspecified' before
               generating.

    y: String. If set, will output Numpy array of specified label column along
       with cleaned dataframe.

    dates: String. If set, will convert column to datetime for new frame.

    linear: Boolean. If false, will create categorical dummies. If true, will
            create boolean dummies (0/1).
    dateform: Datetime parsable string for formatting date. Default is US
              standard

    str_to_num: List of columns. Columns will be treated as strings, and
                converted to numeric where possible. Strips percent signs and
                any other specified characters. Regex functionality to be added
                in future versions.

    symbols: String, or list of strings, of symbols to search and strip when
             converting to numeric. Default is '$'.

    Returns: Cleaned dataframe and either False, or value to be predicted as
             Numpy array.
    """
    n = df.shape[0]
    used = []
    ind = np.array(xrange(n))

    """
    This sequence of if statements will run through all of the functions to set
    up the result dataframe.
    """

    if fields:
        resdf = df[fields]
        used.append('fields')
    else:
        resdf = pd.Dataframe(index = ind)

    if dummys:
        dummies = dummygen(df, dummys, linear, threshold)
        used.append('dummys')
    if y:
        y = df[y].as_matrix()

    if dates:
        used.append('dates')
        resdf[dates] = pd.to_datetime(df[dates], format = dateform)

    s = kwargs.get('str_to_num')
    if s:
        used.append('nums')
        for column in s:
            resdf[column] = numclean(df[column], kwargs.get('symbols', '$'))

    if not used:
        print 'Why did you even run this function?'

    return resdf, y

def numclean(col, symbols = '$'):
    for sym in symbols:
        col = col.str.replace(sym,'')

    if col.str.find('%').mean() == -1:
        return col.astype(float)
    else:
        col = col.str.replace('%', '')
        return col.astype(float)/100



def dummygen(df, dummys, linear = False, threshold = 0):
    n = df.shape[0]
    resdf = pd.Dataframe(index = xrange(n))

    if type(threshold) != int:
        if len(threshold) != len(dummys):
            print 'Error: Threshold count does not match Dummy count'
            return

    if linear:
        for dum in dummys:
            resdf.concat((resdf, pd.get_dummies(df[dum], prefix = dum, dummy_na = 1, drop_first = 1)), axis = 1)
    else:
        for dum in dummys:
            cats = np.zeros(n)
            col = df[dum].as_matrix()
            uni = np.unique(col)
            for key, value in enumerate(uni):
                cats[col == value] = key
            resdf[(dum + '_categorical')] = cats

    return resdf

def norml(column):
    return (column - column.mean())/column.std()
