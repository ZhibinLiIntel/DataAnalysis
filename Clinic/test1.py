# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing

def loadDataSet(path):  # 提取文件中的数据
    data = pd.read_csv(path, sep=',')
    #data = data.dropna(axis = 1, how = 'any')
    data = data[['GENDER_CODE','PSN_AGE','AMT_1_TOTAL','AMT_TOTAL','SELF_PAY_AMT_TOTAL','FUND_AMT','VISIT_TYPE_INBED_CODE','PSN_ACC_AMT_I','CHECK_CNT_I','OUT_LIMIT_AMT_TOTAL_I','AMT_1_TOTAL_I','AMT_3_TOTAL_I','AMT_5_TOTAL_I','CASH_AMT_I','AMT_4_TOTAL_I','FUND_RANGE_AMT_I','SELF_PAY_AMT_TOTAL_I', 'Y_FLAG']]
    return data



def dataDisposeCut(data):
    def dataDiscretize(dataSet, bin, prefix):
        dataSet = pd.cut(dataSet, bins=bin)
        dummies_dataSet = pd.get_dummies(dataSet, prefix=prefix)
        return dummies_dataSet
    test = dataDiscretize(data['GENDER_CODE'], bin = 2, prefix = 'GENDER_CODE')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['PSN_AGE'], bin = 8, prefix = 'PSN_AGE')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['AMT_1_TOTAL'], bin = 8, prefix = 'AMT_1_TOTAL')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['AMT_TOTAL'], bin=8, prefix='AMT_TOTAL')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['SELF_PAY_AMT_TOTAL'], bin=8, prefix='SELF_PAY_AMT_TOTAL')
    data = data.join(test, how='outer')

    #6分类:0和5类等距
    test = dataDiscretize(data['FUND_AMT'], bin=5, prefix='FUND_AMT')
    series = pd.DataFrame(range(len(data['FUND_AMT'])), columns=['FUND_AMT_0'])
    for i in range(len(data['FUND_AMT'])):
        if data.ix[i, 'FUND_AMT'] == 0:
            series.ix[i] = 1
        else:
            series.ix[i] = 0
    data = data.join(series, how='outer')
    data = data.join(test, how='outer')

    #2分类:0和非0
    series = pd.DataFrame(range(len(data['PSN_ACC_AMT_I'])), columns=['PSN_ACC_AMT_I_0'])
    for i in range(len(data['PSN_ACC_AMT_I'])):
        if data.ix[i, 'PSN_ACC_AMT_I'] == 0:
            series.ix[i] = 1
        else:
            series.ix[i] = 0
    data = data.join(series, how='outer')

    test = dataDiscretize(data['CHECK_CNT_I'], bin=3, prefix='CHECK_CNT_I')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['OUT_LIMIT_AMT_TOTAL_I'], bin=8, prefix='OUT_LIMIT_AMT_TOTAL_I')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['AMT_1_TOTAL_I'], bin=3, prefix='AMT_1_TOTAL_I')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['AMT_3_TOTAL_I'], bin=3, prefix='AMT_3_TOTAL_I')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['AMT_5_TOTAL_I'], bin=3, prefix='AMT_5_TOTAL_I')
    data = data.join(test, how='outer')

    #3分类:0和2分类等距
    test = dataDiscretize(data['CASH_AMT_I'], bin=2, prefix='CASH_AMT_I')
    series = pd.DataFrame(range(len(data['CASH_AMT_I'])), columns=['CASH_AMT_I_0'])
    for i in range(len(data['CASH_AMT_I'])):
        if data.ix[i, 'CASH_AMT_I'] == 0:
            series.ix[i] = 1
        else:
            series.ix[i] = 0
    data = data.join(series, how='outer')
    data = data.join(test, how='outer')

    test = dataDiscretize(data['AMT_4_TOTAL_I'], bin=3, prefix='AMT_4_TOTAL_I')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['FUND_RANGE_AMT_I'], bin=3, prefix='FUND_RANGE_AMT_I')
    data = data.join(test, how='outer')
    test = dataDiscretize(data['SELF_PAY_AMT_TOTAL_I'], bin=3, prefix='SELF_PAY_AMT_TOTAL_I')
    data = data.join(test, how='outer')
    data.drop(['GENDER_CODE', 'PSN_AGE', 'AMT_1_TOTAL', 'AMT_TOTAL', 'SELF_PAY_AMT_TOTAL', 'FUND_AMT', 'PSN_ACC_AMT_I', 'CHECK_CNT_I', 'OUT_LIMIT_AMT_TOTAL_I', 'AMT_1_TOTAL_I', 'AMT_3_TOTAL_I', 'AMT_5_TOTAL_I', 'CASH_AMT_I', 'AMT_4_TOTAL_I', 'FUND_RANGE_AMT_I', 'SELF_PAY_AMT_TOTAL_I'], inplace = True, axis = 1)

    for i in range(len(data['Y_FLAG'])):
        if data.ix[i, 'Y_FLAG'] == 'high':
            data.ix[i, 'Y_FLAG'] = 1
        else:
            data.ix[i, 'Y_FLAG'] = 0

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data.drop(['Y_FLAG'], axis = 1, inplace = False), data['Y_FLAG'], random_state=0, train_size=0.7)

    return x_train, x_test, y_train, y_test

def dataDisposeScaler(data):
    for i in range(len(data['Y_FLAG'])):
        if data.ix[i, 'Y_FLAG'] == 'high':
            data.ix[i, 'Y_FLAG'] = 1
        else:
            data.ix[i, 'Y_FLAG'] = 0
    data = data.dropna(how = 'any')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data.drop(['Y_FLAG'], axis = 1, inplace = False), data['Y_FLAG'], random_state=0, train_size=0.7)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    path = 'C:\\Users\\brmt0\\Desktop\\BigData\\data_utf-8.csv'
    data = loadDataSet(path)

#以下两种数据预处理函数二选一，上面为分箱处理，下面为不分箱但标准化
    #x_train, x_test, y_train, y_test = dataDisposeCut(data)
    x_train, x_test, y_train, y_test = dataDisposeScaler(data)

    from sklearn.model_selection import GridSearchCV
    models = [('逻辑回归', GridSearchCV(LogisticRegression(), param_grid={"C": [0.1, 1, 10], "penalty": ['l2', 'l1']}, cv=5)),
              ('决策树', GridSearchCV(tree.DecisionTreeClassifier(), param_grid={"max_depth": [40, 50, 60]}, cv=5)),
              ('AdaBoost', GridSearchCV(AdaBoostClassifier(), param_grid={"learning_rate": [1, 0.5, 0.1], "n_estimators": [40, 50, 60]}, cv=5))]

    '''#测试代码，实验可删掉
    clf = LogisticRegression().fit(X = x_train, y = list(y_train))
    pred = clf.predict(x_test)
    #print pd.DataFrame(pred)
    score = clf.score(x_test, list(y_test))
    print score, pred
    '''

    # 遍历所有模型

    print ("不同算法的准确度分别是：")
    for model in models:
        # 模型训练
        model[1].fit(X = x_train, y = list(y_train))

        # 预测
        #pred = model[1].predict(x_test)
        score = model[1].score(x_test, list(y_test))
        # 输出准确率
        print ("%s: %f" % (model[0], score))
        print ("最佳参数: %s" % (model[1].best_params_))
