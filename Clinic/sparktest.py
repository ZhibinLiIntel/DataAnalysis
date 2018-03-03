import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from pandas import Series
import matplotlib.pyplot as plt
from sklearn import metrics as m
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression



def filter(data):
    df_train = pd.DataFrame(data,columns=['SELF_PAY_AMT_TOTAL_I',\
                                          'FUND_RANGE_AMT_I',\
                                          'AMT_4_TOTAL_I',\
                                          'CASH_AMT_I',\
                                          'AMT_5_TOTAL_I',\
                                          'AMT_3_TOTAL_I',\
                                          'AMT_1_TOTAL_I',\
                                          'OUT_LIMIT_AMT_TOTAL_I',\
                                          'CHECK_CNT_I',\
                                          'PSN_ACC_AMT_I',\
                                          'VISIT_TYPE_INBED_CODE',\
                                          'FUND_AMT',\
                                          'SELF_PAY_AMT_TOTAL',\
                                          'AMT_TOTAL',\
                                          'AMT_1_TOTAL',\
                                          'PSN_AGE',\
                                          'GENDER_CODE',\
                                          'Y_FLAG'])
    return df_train

def binning(data):
    X24 = pd.DataFrame(pd.qcut(data['SELF_PAY_AMT_TOTAL_I'],3,labels=['1','2','3']))
    X22 = pd.DataFrame(pd.qcut(data['FUND_RANGE_AMT_I'],3,labels=['1','2','3']))
    X29 = pd.DataFrame(pd.qcut(data['AMT_4_TOTAL_I'],3,labels=['1','2','3']))
    X20 = pd.DataFrame(pd.qcut(data.loc[data['CASH_AMT_I'] != 0,'CASH_AMT_I'],2,labels=['1','2']))
    X30 = pd.DataFrame(pd.qcut(data['AMT_5_TOTAL_I'],3,labels=['1','2','3']))
    X28 = pd.DataFrame(pd.qcut(data['AMT_3_TOTAL_I'],3,labels=['1','2','3']))
    X26 = pd.DataFrame(pd.qcut(data['AMT_1_TOTAL_I'],3,labels=['1','2','3']))
    X25 = pd.DataFrame(pd.qcut(data['OUT_LIMIT_AMT_TOTAL_I'],8,labels=['1','2','3','4','5','6','7','8']))
    X21 = pd.DataFrame(pd.qcut(data['CHECK_CNT_I'],3,labels=['1','2','3']))
    data.loc[data['PSN_ACC_AMT_I'] == 0,'PSN_ACC_AMT_I'] = 0
    data.loc[data['PSN_ACC_AMT_I'] != 0,'PSN_ACC_AMT_I'] = 1
    X4 = pd.DataFrame(pd.qcut(data.loc[data['FUND_AMT'] != 0, 'FUND_AMT'],5,labels=['1','2','3','4','5']))
    X10 = pd.DataFrame(pd.qcut(data.loc[data['SELF_PAY_AMT_TOTAL'] != 0,'SELF_PAY_AMT_TOTAL'],8,labels=['1','2','3','4','5','6','7','8']))
    #X10 = pd.DataFrame(pd.qcut(data['SELF_PAY_AMT_TOTAL'],9,duplicates='drop'))
    X9 = pd.DataFrame(pd.qcut(data['AMT_TOTAL'],8,labels=['1','2','3','4','5','6','7','8']))
    X12 = pd.DataFrame(pd.qcut(data['AMT_1_TOTAL'],8,labels=['1','2','3','4','5','6','7','8']))
    X2 = pd.DataFrame(pd.qcut(data['PSN_AGE'],8,labels=['1','2','3','4','5','6','7','8']))
    test = pd.concat([X24,X22,X29,X20,X30,X28,X26,X25,X21,data['PSN_ACC_AMT_I'],data['VISIT_TYPE_INBED_CODE'],X4,X10,X9,X12,X2,data['GENDER_CODE'],data['Y_FLAG']],axis=1)
    test['CASH_AMT_I'] = test['CASH_AMT_I'].cat.add_categories([0]);
    test['CASH_AMT_I'].fillna(0, inplace=True)
    test['SELF_PAY_AMT_TOTAL'] = test['SELF_PAY_AMT_TOTAL'].cat.add_categories([0]);
    test['SELF_PAY_AMT_TOTAL'].fillna(0, inplace=True)
    test['FUND_AMT'] = test['FUND_AMT'].cat.add_categories([0]);
    test['FUND_AMT'].fillna(0, inplace=True)
    return test
    #data.to_csv('C:\\Users\\brmt0\\Desktop\\BigData\\test.csv',index=False)

def getWOE(data):
    
    
    high = data.loc[data['Y_FLAG'] == 'high','GENDER_CODE']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'GENDER_CODE']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    male_woe = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    female_woe = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data.loc[data['GENDER_CODE'] == 1, 'GENDER_CODE'] = male_woe
    data.loc[data['GENDER_CODE'] == 2, 'GENDER_CODE'] = female_woe
      
        
    high = data.loc[data['Y_FLAG'] == 'high','SELF_PAY_AMT_TOTAL_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'SELF_PAY_AMT_TOTAL_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X24_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X24_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X24_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['SELF_PAY_AMT_TOTAL_I'] = data['SELF_PAY_AMT_TOTAL_I'].astype('float64')
    data.loc[data['SELF_PAY_AMT_TOTAL_I'] == 1, 'SELF_PAY_AMT_TOTAL_I'] = X24_woe1
    data.loc[data['SELF_PAY_AMT_TOTAL_I'] == 2, 'SELF_PAY_AMT_TOTAL_I'] = X24_woe2
    data.loc[data['SELF_PAY_AMT_TOTAL_I'] == 3, 'SELF_PAY_AMT_TOTAL_I'] = X24_woe3
    #print(X24_woe1,X24_woe2,X24_woe3)
    #print(data['SELF_PAY_AMT_TOTAL_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','FUND_RANGE_AMT_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'FUND_RANGE_AMT_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X22_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X22_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X22_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['FUND_RANGE_AMT_I'] = data['FUND_RANGE_AMT_I'].astype('float64')
    data.loc[data['FUND_RANGE_AMT_I'] == 1, 'FUND_RANGE_AMT_I'] = X22_woe1
    data.loc[data['FUND_RANGE_AMT_I'] == 2, 'FUND_RANGE_AMT_I'] = X22_woe2
    data.loc[data['FUND_RANGE_AMT_I'] == 3, 'FUND_RANGE_AMT_I'] = X22_woe3
    #print(X22_woe1,X22_woe2,X22_woe3)
    #print(data['FUND_RANGE_AMT_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','AMT_4_TOTAL_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'AMT_4_TOTAL_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X29_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X29_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X29_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['AMT_4_TOTAL_I'] = data['AMT_4_TOTAL_I'].astype('float64')
    data.loc[data['AMT_4_TOTAL_I'] == 1, 'AMT_4_TOTAL_I'] = X29_woe1
    data.loc[data['AMT_4_TOTAL_I'] == 2, 'AMT_4_TOTAL_I'] = X29_woe2
    data.loc[data['AMT_4_TOTAL_I'] == 3, 'AMT_4_TOTAL_I'] = X29_woe3
    
    
    high = data.loc[data['Y_FLAG'] == 'high','CASH_AMT_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'CASH_AMT_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X20_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X20_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X20_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['CASH_AMT_I'] = data['CASH_AMT_I'].astype('float64')
    data.loc[data['CASH_AMT_I'] == 0, 'CASH_AMT_I'] = X20_woe1
    data.loc[data['CASH_AMT_I'] == 1, 'CASH_AMT_I'] = X20_woe2
    data.loc[data['CASH_AMT_I'] == 2, 'CASH_AMT_I'] = X20_woe3
    #print(data['CASH_AMT_I'])
    #print(X20_woe1,X20_woe2,X20_woe3)


    high = data.loc[data['Y_FLAG'] == 'high','AMT_5_TOTAL_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'AMT_5_TOTAL_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X30_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X30_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X30_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['AMT_5_TOTAL_I'] = data['AMT_5_TOTAL_I'].astype('float64')
    data.loc[data['AMT_5_TOTAL_I'] == 1, 'AMT_5_TOTAL_I'] = X30_woe1
    data.loc[data['AMT_5_TOTAL_I'] == 2, 'AMT_5_TOTAL_I'] = X30_woe2
    data.loc[data['AMT_5_TOTAL_I'] == 3, 'AMT_5_TOTAL_I'] = X30_woe3
 
    
    high = data.loc[data['Y_FLAG'] == 'high','AMT_3_TOTAL_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'AMT_3_TOTAL_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X28_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X28_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X28_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['AMT_3_TOTAL_I'] = data['AMT_3_TOTAL_I'].astype('float64')
    data.loc[data['AMT_3_TOTAL_I'] == 1, 'AMT_3_TOTAL_I'] = X28_woe1
    data.loc[data['AMT_3_TOTAL_I'] == 2, 'AMT_3_TOTAL_I'] = X28_woe2
    data.loc[data['AMT_3_TOTAL_I'] == 3, 'AMT_3_TOTAL_I'] = X28_woe3
    #print(data['AMT_3_TOTAL_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','AMT_1_TOTAL_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'AMT_1_TOTAL_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X26_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X26_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X26_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['AMT_1_TOTAL_I'] = data['AMT_1_TOTAL_I'].astype('float64')
    data.loc[data['AMT_1_TOTAL_I'] == 1, 'AMT_1_TOTAL_I'] = X26_woe1
    data.loc[data['AMT_1_TOTAL_I'] == 2, 'AMT_1_TOTAL_I'] = X26_woe2
    data.loc[data['AMT_1_TOTAL_I'] == 3, 'AMT_1_TOTAL_I'] = X26_woe3
    #print(data['AMT_1_TOTAL_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','OUT_LIMIT_AMT_TOTAL_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'OUT_LIMIT_AMT_TOTAL_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X25_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X25_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X25_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X25_woe4 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X25_woe5 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X25_woe6 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X25_woe7 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X25_woe8 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['OUT_LIMIT_AMT_TOTAL_I'] = data['OUT_LIMIT_AMT_TOTAL_I'].astype('float64')
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 1, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe1
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 2, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe2
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 3, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe3
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 4, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe4
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 5, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe5
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 6, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe6
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 7, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe7
    data.loc[data['OUT_LIMIT_AMT_TOTAL_I'] == 8, 'OUT_LIMIT_AMT_TOTAL_I'] = X25_woe8
    #print(data['OUT_LIMIT_AMT_TOTAL_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','CHECK_CNT_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'CHECK_CNT_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X21_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X21_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X21_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['CHECK_CNT_I'] = data['CHECK_CNT_I'].astype('float64')
    data.loc[data['CHECK_CNT_I'] == 1, 'CHECK_CNT_I'] = X21_woe1
    data.loc[data['CHECK_CNT_I'] == 2, 'CHECK_CNT_I'] = X21_woe2
    data.loc[data['CHECK_CNT_I'] == 3, 'CHECK_CNT_I'] = X21_woe3
    #print(data['CHECK_CNT_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','PSN_ACC_AMT_I']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'PSN_ACC_AMT_I']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X17_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X17_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    data['PSN_ACC_AMT_I'] = data['PSN_ACC_AMT_I'].astype('float64')
    data.loc[data['PSN_ACC_AMT_I'] == 0, 'PSN_ACC_AMT_I'] = X17_woe1
    data.loc[data['PSN_ACC_AMT_I'] != 0, 'PSN_ACC_AMT_I'] = X17_woe2
    #print(data['PSN_ACC_AMT_I'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','VISIT_TYPE_INBED_CODE']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'VISIT_TYPE_INBED_CODE']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X21_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X21_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X21_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    #data['VISIT_TYPE_INBED_CODE'] = data['VISIT_TYPE_INBED_CODE'].astype('float64')
    data.loc[data['VISIT_TYPE_INBED_CODE'] == 0, 'VISIT_TYPE_INBED_CODE'] = X21_woe1
    data.loc[data['VISIT_TYPE_INBED_CODE'] == 1, 'VISIT_TYPE_INBED_CODE'] = X21_woe2
    data.loc[data['VISIT_TYPE_INBED_CODE'] == 2, 'VISIT_TYPE_INBED_CODE'] = X21_woe3
    #print(data['VISIT_TYPE_INBED_CODE'])
   
    
    high = data.loc[data['Y_FLAG'] == 'high','FUND_AMT']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'FUND_AMT']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X4_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X4_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X4_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X4_woe4 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X4_woe5 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X4_woe6 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['FUND_AMT'] = data['FUND_AMT'].astype('float64')
    data.loc[data['FUND_AMT'] == 0, 'FUND_AMT'] = X4_woe1
    data.loc[data['FUND_AMT'] == 1, 'FUND_AMT'] = X4_woe2
    data.loc[data['FUND_AMT'] == 2, 'FUND_AMT'] = X4_woe3
    data.loc[data['FUND_AMT'] == 3, 'FUND_AMT'] = X4_woe4
    data.loc[data['FUND_AMT'] == 4, 'FUND_AMT'] = X4_woe5
    data.loc[data['FUND_AMT'] == 5, 'FUND_AMT'] = X4_woe6
    #print(data['FUND_AMT'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','SELF_PAY_AMT_TOTAL']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'SELF_PAY_AMT_TOTAL']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X10_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X10_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X10_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X10_woe4 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X10_woe5 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X10_woe6 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X10_woe7 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X10_woe8 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['SELF_PAY_AMT_TOTAL'] = data['SELF_PAY_AMT_TOTAL'].astype('float64')
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 1, 'SELF_PAY_AMT_TOTAL'] = X10_woe1
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 2, 'SELF_PAY_AMT_TOTAL'] = X10_woe2
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 3, 'SELF_PAY_AMT_TOTAL'] = X10_woe3
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 4, 'SELF_PAY_AMT_TOTAL'] = X10_woe4
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 5, 'SELF_PAY_AMT_TOTAL'] = X10_woe5
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 6, 'SELF_PAY_AMT_TOTAL'] = X10_woe6
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 7, 'SELF_PAY_AMT_TOTAL'] = X10_woe7
    data.loc[data['SELF_PAY_AMT_TOTAL'] == 8, 'SELF_PAY_AMT_TOTAL'] = X10_woe8
    #print(data['SELF_PAY_AMT_TOTAL'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','AMT_TOTAL']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'AMT_TOTAL']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X9_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X9_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X9_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X9_woe4 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X9_woe5 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X9_woe6 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X9_woe7 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X9_woe8 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['AMT_TOTAL'] = data['AMT_TOTAL'].astype('float64')
    data.loc[data['AMT_TOTAL'] == 1, 'AMT_TOTAL'] = X9_woe1
    data.loc[data['AMT_TOTAL'] == 2, 'AMT_TOTAL'] = X9_woe2
    data.loc[data['AMT_TOTAL'] == 3, 'AMT_TOTAL'] = X9_woe3
    data.loc[data['AMT_TOTAL'] == 4, 'AMT_TOTAL'] = X9_woe4
    data.loc[data['AMT_TOTAL'] == 5, 'AMT_TOTAL'] = X9_woe5
    data.loc[data['AMT_TOTAL'] == 6, 'AMT_TOTAL'] = X9_woe6
    data.loc[data['AMT_TOTAL'] == 7, 'AMT_TOTAL'] = X9_woe7
    data.loc[data['AMT_TOTAL'] == 8, 'AMT_TOTAL'] = X9_woe8
    #print(data['AMT_TOTAL'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','AMT_1_TOTAL']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'AMT_1_TOTAL']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X12_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X12_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X12_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X12_woe4 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X12_woe5 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X12_woe6 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X12_woe7 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X12_woe8 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['AMT_1_TOTAL'] = data['AMT_1_TOTAL'].astype('float64')
    data.loc[data['AMT_1_TOTAL'] == 1, 'AMT_1_TOTAL'] = X12_woe1
    data.loc[data['AMT_1_TOTAL'] == 2, 'AMT_1_TOTAL'] = X12_woe2
    data.loc[data['AMT_1_TOTAL'] == 3, 'AMT_1_TOTAL'] = X12_woe3
    data.loc[data['AMT_1_TOTAL'] == 4, 'AMT_1_TOTAL'] = X12_woe4
    data.loc[data['AMT_1_TOTAL'] == 5, 'AMT_1_TOTAL'] = X12_woe5
    data.loc[data['AMT_1_TOTAL'] == 6, 'AMT_1_TOTAL'] = X12_woe6
    data.loc[data['AMT_1_TOTAL'] == 7, 'AMT_1_TOTAL'] = X12_woe7
    data.loc[data['AMT_1_TOTAL'] == 8, 'AMT_1_TOTAL'] = X12_woe8
    #print(data['AMT_1_TOTAL'])
    
    
    high = data.loc[data['Y_FLAG'] == 'high','PSN_AGE']
    high_ret = pd.value_counts(high)
    high_total = high.count()
    low = data.loc[data['Y_FLAG'] == 'low', 'PSN_AGE']
    low_ret = pd.value_counts(low)
    low_total = low.count()
    X2_woe1 = math.log((high_ret[0]/high_total) / (low_ret[0]/low_total))
    X2_woe2 = math.log((high_ret[1]/high_total) / (low_ret[1]/low_total))
    X2_woe3 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X2_woe4 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X2_woe5 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X2_woe6 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X2_woe7 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    X2_woe8 = math.log((high_ret[2]/high_total) / (low_ret[2]/low_total))
    data['PSN_AGE'] = data['PSN_AGE'].astype('float64')
    data.loc[data['PSN_AGE'] == 1, 'PSN_AGE'] = X2_woe1
    data.loc[data['PSN_AGE'] == 2, 'PSN_AGE'] = X2_woe2
    data.loc[data['PSN_AGE'] == 3, 'PSN_AGE'] = X2_woe3
    data.loc[data['PSN_AGE'] == 4, 'PSN_AGE'] = X2_woe4
    data.loc[data['PSN_AGE'] == 5, 'PSN_AGE'] = X2_woe5
    data.loc[data['PSN_AGE'] == 6, 'PSN_AGE'] = X2_woe6
    data.loc[data['PSN_AGE'] == 7, 'PSN_AGE'] = X2_woe7
    data.loc[data['PSN_AGE'] == 8, 'PSN_AGE'] = X2_woe8
    #print(data['PSN_AGE'])
    
    
    data.loc[data['Y_FLAG'] == 'high','Y_FLAG'] = 1
    data.loc[data['Y_FLAG'] == 'low', 'Y_FLAG'] = 0
    return data
    
if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\brmt0\\Desktop\\BigData\\data_utf-8.csv')
    train = filter(df)
    train = binning(train)
    test = getWOE(train)
    test = test.dropna()
    data_x = test.drop(['Y_FLAG'],axis=1)
    data_y = pd.DataFrame(test['Y_FLAG'])
    
    
    classifiers = {'Adaptive Boosting Classifier':AdaBoostClassifier(),'Logistic Regression':LogisticRegression(),'DecisionTree':tree.DecisionTreeClassifier()}
    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)
    rs = ShuffleSplit(n_splits=1, test_size=0.3,random_state=0)
    rs.get_n_splits(data_x,data_y)
    
    for Name,classify in classifiers.items():
        for train_index, test_index in rs.split(data_x,data_y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X,X_test = data_x.iloc[train_index], data_x.iloc[test_index]
            y,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
            cls = classify
            cls =cls.fit(X,y.values.ravel())
            y_out = cls.predict(X_test)
            accuracy = m.accuracy_score(y_test,y_out)
            precision = m.precision_score(y_test,y_out,average='macro')
            recall = m.recall_score(y_test,y_out,average='macro')
            f1_score = m.f1_score(y_test,y_out,average='macro')
            roc_auc = roc_auc_score(y_out,y_test)
            log_entry = pd.DataFrame([[Name,accuracy]], columns=log_cols)
            #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
            log = log.append(log_entry)
            #metric = metric.append(metric_entry)
    print(log)
    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log)  
    plt.show()
    
    #print(test.info())
    
    
