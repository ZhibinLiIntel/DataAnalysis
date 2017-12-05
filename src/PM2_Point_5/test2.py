#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import unicodecsv
import seaborn as sb
from scipy.interpolate import spline
from matplotlib.pyplot import ylabel
from statsmodels.sandbox.nonparametric.kdecovclass import plotkde
from sklearn.cluster import KMeans 

df = pd.read_csv("C:\\KnownSec\\PM2.5\\PRSA_data_2010.1.1-2014.12.31.csv")
df.columns=['num','year','month','day','hour','value','dewp','temp','pres','cbwd','lws','ls','lr']

df.dropna(inplace=True)
df_test = pd.DataFrame(df.ix[:,['value','temp']])
print(df_test)

k=100
iteration = 500
data = pd.read_excel("C:\\KnownSec\\PM2.5\\testfile.xls")
kmodel = KMeans(n_clusters=5)
y_pred = kmodel.fit_predict(data)

x = data.ix[:,'value']    
print(x) 
y = data.ix[:,'temp']  
print(y)    
  
for i in range(k):
    plt.scatter(x,y)

plt.show()
r1 = pd.Series(kmodel.labels_).value_counts()  #统计各个类别的数目
r2 = pd.DataFrame(kmodel.cluster_centers_)     #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + [u'类别数目'] #重命名表头
print(r)

r = pd.concat([data, pd.Series(kmodel.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(data.columns) + [u'聚类类别'] #重命名表头



#df_test.to_excel("C:\\KnownSec\\PM2.5\\testfile.xls")