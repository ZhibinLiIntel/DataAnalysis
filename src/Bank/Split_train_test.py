#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('C:\\KnownSec\\Bank\\bank-full2.csv')
df.info()
print(df.describe())

rs=StratifiedShuffleSplit(n_splits=1,train_size=0.90,test_size=0.10,random_state=0)
train_set=np.array(pd.DataFrame(df.ix[:,0:15]))
test_set=np.array(pd.DataFrame(df.ix[:,'y']))

for train_index, test_index in rs.split(train_set, test_set):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_set[train_index], train_set[test_index]
    y_train, y_test = test_set[train_index], test_set[test_index]

df_test = pd.DataFrame(X_test,index=None,columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous'])
df_y_test = pd.DataFrame(y_test,columns=['y'])
df_test = pd.concat([df_test,df_y_test],axis=1)
df_test.to_csv('C:\\KnownSec\\Bank\\test2.csv',index=False)

df_train = pd.DataFrame(X_train,columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous'])
df_y_train = pd.DataFrame(y_train,columns=['y'])
df_train = pd.concat([df_train,df_y_train],axis=1)
df_train.to_csv('C:\\KnownSec\\Bank\\train2.csv',index=False)