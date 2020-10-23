import numpy as np
import pandas as pd
from time import time
import scipy.io as sio
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from L1_Matine import elasticNet, lassodimension,lassolarsdimension
from L1_Matine import selectFromLinearSVC,selectFromExtraTrees,logistic_dimension
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
#from LR_test import logistic_LR
import utils.tools as utils


row=data.shape[0]
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
data_=data[index,:]
shu=data_[:,np.array(range(1,column))]
shu=scale(shu)
label=data_[:,0]
label[label==0]=-1

X=shu
label[label==-1]=0
y=label
num_class=2
loo = LeaveOneOut()
sepscores = []
y_score=np.ones((1,2))*0.5
y_class=np.ones((1,1))*0.5       
for train, test in loo.split(X): 
    cv_clf =  GaussianNB()
    X_train=X[train]
    y_train=y[train] 
    X_test=X[test]
    y_test=y[test]
    y_sparse=utils.to_categorical(y)
    y_train_sparse=utils.to_categorical(y_train)
    y_test_sparse=utils.to_categorical(y_test)
    hist=cv_clf.fit(X_train, y_train)
    y_predict_score=cv_clf.predict_proba(X_test) 
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
y_sparse=utils.to_categorical(y)
yscore_sum = pd.DataFrame(data=y_score)
yscore_sum.to_csv('yscore_NB_1075_knife_no.csv')
ytest_sum = pd.DataFrame(data=y_sparse)
ytest_sum.to_csv('ytest_NB_1075_knife_no.csv')
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
auc_score=result[6]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='NB ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('NB_1075_knife_no.csv')






	
	