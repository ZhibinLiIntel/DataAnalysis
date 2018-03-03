import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics as m
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


classifiers = {'Adaptive Boosting Classifier':AdaBoostClassifier(),'Logistic Regression':LogisticRegression(),'Random Forest Classifier': RandomForestClassifier()}


df = pd.read_csv('C:\\KnownSec\\Bank\\pro_test2.csv')
df2 = pd.read_csv('C:\\KnownSec\\Bank\\pro_test2.csv')

test_y = pd.DataFrame(df2['y'])
test_X = df.drop(['y'],axis=1)
data_y = pd.DataFrame(df['y'])
data_X = df.drop(['y'],axis=1)
print(data_X.columns)
log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]

log = pd.DataFrame(columns=log_cols)

import warnings
warnings.filterwarnings('ignore')
rs = ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
rs.get_n_splits(data_X,data_y)
for Name,classify in classifiers.items():
    for train_index, test_index in rs.split(data_X,data_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        sm = SMOTE(random_state=12, ratio = 1.0)
        X,X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        X,y = sm.fit_sample(X, y)
        cls = classify
        cls =cls.fit(X,y)
        y_out = cls.predict(X_test)
        accuracy = m.accuracy_score(y_test,y_out)
        precision = m.precision_score(y_test,y_out,average='macro')
        recall = m.recall_score(y_test,y_out,average='macro')
        f1_score = m.f1_score(y_test,y_out,average='macro')
        roc_auc = roc_auc_score(y_out,y_test)
        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)
        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
        log = log.append(log_entry)
        #metric = metric.append(metric_entry)
'''
rs = ShuffleSplit(n_splits=1, test_size=0.1,random_state=0)
rs.get_n_splits(test_X,test_y)
for Name,classify in classifiers.items():
    for train2_index, test2_index in rs.split(test_X,test_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        sm = SMOTE(random_state=12, ratio = 1.0)
        X2,X2_test = test_X.iloc[train2_index], test_X.iloc[test2_index]
        y2,y2_test = test_y.iloc[train2_index], test_y.iloc[test2_index]
        X2,y2 = sm.fit_sample(X2, y2)
        cls2 = classify
        cls2 =cls2.fit(X2,y2)
        y2_out = cls2.predict(X2_test)
        accuracy = m.accuracy_score(y2_test,y2_out)
        precision = m.precision_score(y2_test,y2_out,average='macro')
        recall = m.recall_score(y2_test,y2_out,average='macro')
        f1_score = m.f1_score(y2_test,y2_out,average='macro')
        roc_auc = roc_auc_score(y2_out,y2_test)
        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)
        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
        log = log.append(log_entry)
        #metric = metric.append(metric_entry)
'''
print(log)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log)  
plt.show()