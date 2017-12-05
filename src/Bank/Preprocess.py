import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from _codecs import encode

df = pd.read_csv('C:\\KnownSec\\Bank\\test2.csv')
df.info()

def age_trans(data):
    data.loc[data['age'] == 1,'age'] = 'adult'
    data.loc[data['age'] == 2,'age'] = 'middleaged'
    data.loc[data['age'] == 3,'age'] = 'old'
    return data
    
def age_(data):
    data.loc[(data['age'] <= 35) & (data['age'] >= 18),'age'] = 1
    data.loc[(data['age'] <= 60) & (data['age'] >= 36),'age'] = 2
    #data.loc[(data['age'] <= 60) & (data['age'] >= 46),'Elderly'] = 1
    data.loc[data['age'] >=61,'age'] = 3
    #data = age_trans(data)
    return data

def encode_bin_attrs(data, bin_attrs):  
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1 
    return data

def encode_edu_attrs(data):
    values = ["primary", "secondary", "tertiary"]
    levels = range(1,len(values)+1)
    dict_levels = dict(zip(values, levels))
    data = age_(data)   
    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]    
    return data

def encode_age_attrs(data):
    values = ["adult", "middleaged", "old"]
    levels = range(1,len(values)+1)
    dict_levels = dict(zip(values, levels))    
    for v in values:
        data.loc[data['age'] == v, 'age'] = dict_levels[v]    
    return data

def encode_cate_attrs(data, cate_attrs): 
    #data = encode_edu_attrs(data)
    #cate_attrs.remove('education')
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)    
    return data

def test(data, numeric_attr):
    for k in numeric_attr:
        scaler = preprocessing.StandardScaler()
        data[k] = scaler.fit_transform(data[k])
    return data;
#df = df.drop(['day'],['month'])
for i in [['job'],['marital'],['contact'],['month'],['day']]:
    df = encode_cate_attrs(df, i)

df = encode_edu_attrs(df)
df = age_(df)
df.info()
'''
df = encode_age_attrs(df)
df.info()
'''
for j in [['default'],['housing'],['loan'],['y']]:
    df = encode_bin_attrs(df, j)
    
numeric_attr = [['balance'],['duration'],['campaign'],['pdays'],['previous']]
df = test(df,numeric_attr)

df.to_csv('C:\\KnownSec\\Bank\\pro_test2.csv',index=False)