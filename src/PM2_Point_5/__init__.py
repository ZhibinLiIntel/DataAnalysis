#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import unicodecsv
import seaborn as sb
from scipy.interpolate import spline
from matplotlib.pyplot import ylabel
from sklearn.cluster import KMeans 

df = pd.read_csv("C:\\KnownSec\\PM2.5\\PRSA_data_2010.1.1-2014.12.31.csv")
df.columns=['num','year','month','day','hour','value','dewp','temp','pres','cbwd','lws','ls','lr']

def get_grade(value):
    if value <= 50 and value>=0:
        return 'lv1'
    elif value <= 100:
        return 'Lv2'
    elif value <= 150:
        return 'Lv3'
    elif value <= 200:
        return 'Lv4'
    elif value <= 300:
        return 'Lv5'
    elif value <= 500:
        return 'Lv6'
    elif value > 500:
        return 'Out of range' 
    else:
        return 'unknown'  

#print(df.groupby(['year']).count())

#df.dropna(inplace=True)
#remain to be improved!!!Use groupby function
df_2010 = pd.DataFrame(df.loc[0:8759,:])
df_2011 = pd.DataFrame(df.loc[8760:17519,:])
df_2012 = pd.DataFrame(df.loc[17520:26303,:])
df_2013 = pd.DataFrame(df.loc[26304:35063,:])
df_2014 = pd.DataFrame(df.loc[35064:43824,:])


df_2010.loc[:, 'Grade'] = df_2010['value'].apply(get_grade)
df_2011.loc[:, 'Grade'] = df_2011['value'].apply(get_grade)
df_2012.loc[:, 'Grade'] = df_2012['value'].apply(get_grade)
df_2013.loc[:, 'Grade'] = df_2013['value'].apply(get_grade)
df_2014.loc[:, 'Grade'] = df_2014['value'].apply(get_grade)

grade_2010 = df_2010.groupby(['Grade']).size()/len(df_2010)
grade_2011 = df_2011.groupby(['Grade']).size()/len(df_2011)
grade_2012 = df_2012.groupby(['Grade']).size()/len(df_2012)
grade_2013 = df_2013.groupby(['Grade']).size()/len(df_2013)
grade_2014 = df_2014.groupby(['Grade']).size()/len(df_2014)


x1 = df_2010['num']
y1 = df_2010['value']
y2 = df_2010['dewp']
y3 = df_2010['temp']
y4 = df_2010['pres']
y5 = df_2010['lws']
y6 = df_2010['ls']
y7 = df_2010['lr']
plt.plot(x1,y1)
plt.plot(x1,y2)
plt.plot(x1,y3)
#plt.plot(x1,y4)
plt.plot(x1,y5)
plt.plot(x1,y6)
plt.plot(x1,y7)
#plt.subplot2grid((1,5),(0,0))
#df_month = pd.DataFrame({'2010':df_2010})
df_hour = pd.DataFrame({'month':df_2010.ix[:,'month'],'day':df_2010.ix[:,'day'],'hour':df_2010.ix[:,'hour'],'2010':df_2010.ix[:,'value']})
df_hour = df_hour.merge(df_2011.ix[:,['month','day','hour','value']],on=('month','day','hour'))
df_hour.rename_axis({'value':'2011'}, axis="columns", inplace=True)
print(df_hour.describe())
#sb.kdeplot(df_hour.ix[:,'hour'])
print(df_hour.ix[:,'hour'])
#df_2010.value.plot(kind="kde")
#df_2010.dewp.plot(kind="kde")
#df_2010.temp.plot(kind="kde")
#df_2010.pres.plot(kind="kde")
#df_2010.lws.plot(kind="kde")
#df_hour.ix[:,['2010']].plot(title='2010年pm2.5分布', figsize=(12,4))
#plt.xlabel(u'小时数（从一年的第一个小时开始累计）')
#plt.ylabel(u'pm2.5指数')
plt.xlim(1000,1100)
plt.ylim(-50,200)
plt.legend((u'pm2.5指数', u'露点温度',u'温度',u'累计风速','Cumulated hours of snow','Cumulated hours of rain'),loc='best')

#test_year = df.groupby(['month'])['value'].mean()
#df_year = pd.DataFrame({'test':test_year}, index = np.arange(1,13))

#df_year.ix[:,].plot(title='PM2.5', kind="kde",figsize=(8,4))
#fig = plt.figure()
#fig.set(alpha=0.2)


#df.num[df.year==2011].plot(kind="kde")
#df.num[df.year==2012].plot(kind="kde")
#df.num[df.year==2013].plot(kind="kde")
#df.num[df.year==2014].plot(kind="kde")
plt.show()

