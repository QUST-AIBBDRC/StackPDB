#import tensorflow as tf
import time
import pickle as p
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import scipy.io as sio
from sklearn import svm
from sklearn.svm import SVC
import utils.tools as utils
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression

def get_stacking(clf, x_train, y_train, x_test, num_class,n_folds=5):

    kf = KFold(n_splits=n_folds)
    second_level_train_set=[]
    test_nfolds_set=[]
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        second_level_train_ = clf.predict_proba(x_tst)
        second_level_train_set.append(second_level_train_)
        test_nfolds= clf.predict_proba(x_test)
        test_nfolds_set.append(test_nfolds)   
    train_second=second_level_train_set
    train_second_level=np.concatenate((train_second[0],train_second[1],train_second[2],train_second[3],train_second[4]),axis=0) 
    test_second_level_=np.array(test_nfolds_set)   
    test_second_level=np.mean(test_second_level_,axis = 0)
    return train_second_level,test_second_level
def get_first_level(train_x, train_y, test_x,num_class):
    lgm_model1 = lgb.LGBMClassifier(n_estimators=500)
    xbt_model1 = xgb.XGBClassifier(n_estimators=500)
    lgm_model2 = lgb.LGBMClassifier(n_estimators=500)
    xbt_model2 = xgb.XGBClassifier(n_estimators=500)
    train_sets = []
    test_sets = []
    for clf in [lgm_model1,xbt_model1,lgm_model2,xbt_model2]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x,num_class)
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train = np.concatenate([result_set.reshape(-1,num_class) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,num_class) for y_test_set in test_sets], axis=1)
    return meta_train,meta_test
def get_second_level(train_dim,train_label,test_dim,num_class):
    meta_train,meta_test=get_first_level(train_dim,train_label,test_dim,num_class)
    SVM=SVC(probability=True)
    hist=SVM.fit(meta_train,train_label)
    pre_score=SVM.predict_proba(meta_test)
    return meta_train,meta_test,pre_score 



num_class=2
loo = LeaveOneOut()
sepscores = []
y_score=np.ones((1,2))*0.5
y_class=np.ones((1,1))*0.5       
for train, test in loo.split(X):

    X_train=X[train]
    y_train=y[train] 
    X_test=X[test]
    y_test=y[test]
    y_sparse=utils.to_categorical(y)
    y_train_sparse=utils.to_categorical(y_train)
    y_test_sparse=utils.to_categorical(y_test)
    meta_train,meta_test,y_predict_score=get_second_level(X_train,y_train,X_test,num_class)
    y_predict_class= utils.categorical_probas_to_classes(y_predict_score)
    y_score=np.vstack((y_score,y_predict_score))
    y_class=np.vstack((y_class,y_predict_class))
    cv_clf=[]
y_class=y_class[1:]
y_score=y_score[1:]
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
roc_auc = auc(fpr, tpr)
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y)
result=[acc,precision,npv,sensitivity,specificity,mcc,roc_auc]
row=y_score.shape[0]
#column=data.shape[1]
y_sparse=utils.to_categorical(y)
yscore_sum = pd.DataFrame(data=y_score)
yscore_sum.to_csv('yscore_5_stacking.csv')
ytest_sum = pd.DataFrame(data=y_sparse)
ytest_sum.to_csv('ytest_5_stacking.csv')
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
auc_score=auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='stacking ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('1075_5.csv')


print("acc=%.2f%% " % (result[0]*100))
print("precision=%.2f%% " % (result[1]*100))
print("npv=%.2f%% " % (result[2]*100))
print("sensitivity=%.2f%% " % (result[3]*100))
print("specificity=%.2f%% " % (result[4]*100))
print("mcc=%.2f%%" % (result[5]*100))
print("auc=%.2f%%" % (result[6]*100))


