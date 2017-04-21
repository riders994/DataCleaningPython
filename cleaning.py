import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation as cv
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./Train.csv')

##Creating Initial Test
simpledf = df[['SalesID','ModelID','YearMade','MachineHoursCurrentMeter','UsageBand','saledate','state','fiModelDesc','fiBaseModel','ProductGroup']]
labels = df['SalePrice']
simpledf.dropna(axis=0, inplace = 1)

UsageDums = pd.get_dummies(simpledf['UsageBand'], prefix='Usage', drop_first=1)
