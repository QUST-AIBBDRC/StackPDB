import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from L1_Matine import elasticNet, lassodimension,lassolarsdimension,selectFromLinearSVC
#from L1_Matine import selectFromExtraTrees,logistic_dimension,logistic_LR
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.svm import SVC
from MRMR import mrmr,jmi
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from RFE_feature_selection import SVM_RFE_selection,LOG_RFE_selection,XGB_selection

row=data.shape[0]
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
data_=data[index,:]
shu=data_[:,np.array(range(1,column))]
shu=scale(shu)#
label=data_[:,0]
label[label==0]=-1
data_1,mask1=elasticNet(shu, label)#
data_2,mask2=lassodimension(shu,label)#lasso
data_3,mask3=lassolarsdimension(shu,label)
data_4=selectFromLinearSVC(shu,label,1.5)#
data_6=SVM_RFE_selection(100,shu,label)
data_5=XGB_selection(100,shu,label)
X=data_5
label[label==-1]=0
y=label

