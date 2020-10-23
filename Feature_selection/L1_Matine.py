# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:58:19 2018

@author:
"""
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.preprocessing import normalize,Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoLarsCV,LassoLars
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression

def logistic_dimension(data,label,parameter=0.5):
    logistic_=LogisticRegression(penalty="l1", C=parameter)
    model=SelectFromModel(logistic_)
    new_data=model.fit_transform(data, label)
    mask=model.get_support(indices=True)
    return new_data,mask

def omp_omp(data,label,n_nonzero_coefs=300):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(data, label)
    coef = omp.coef_
    idx_r, = coef.nonzero()
    #omp_cv = OrthogonalMatchingPursuitCV()
    #omp_cv.fit(X, y_noisy)
    #coef = omp_cv.coef_
    #idx_r, = coef.nonzero()
    new_data=data[:,idx_r]
    return new_data,idx_r

def chi2_chi2(data,label,k=300):
    model_chi2= SelectKBest(chi2, k=k)
    new_data=model_chi2.fit_transform(data,label)
    return new_data

def mutual_mutual(data,label,k=300):
    model_mutual= SelectKBest(mutual_info_classif, k=k)
    new_data=model_mutual.fit_transform(data, label)
    return new_data

def elasticNet(data,label,alpha =np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])):
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.4,max_iter=2000).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.4,max_iter=2000)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask
def lassodimension(data,label,alpha=np.array([0.001,0.002,0.003,0.004, 0.005, 0.006, 0.007, 0.008,0.009, 0.01])):#alpha代表想要传递的alpha的一组值,用在循环中,来找出一个尽可能好的alpha的值
    lassocv=LassoCV(cv=5, alphas=alpha,max_iter=2000).fit(data, label)
#    lasso = Lasso(alpha=lassocv.alpha_, random_state=0)#
    x_lasso = lassocv.fit(data,label)#
    mask = x_lasso.coef_ != 0 
    new_data = data[:,mask]  
    return new_data,mask 
def lassolarsdimension(data,label):
    lassolarscv=LassoLarsCV(cv=5,max_iter=400).fit(data, label)
    lassolars = LassoLars(alpha=lassolarscv.alpha_)
    x_lassolars = lassolars.fit(data,label)
    mask = x_lassolars.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask

def selectFromLinearSVC(data,label,lamda):
    lsvc = LinearSVC(C=lamda, penalty="l1", dual=False).fit(data,label)
    model = SelectFromModel(lsvc,prefit=True)
    new_data= model.transform(data)
    return new_data

def selectFromExtraTrees(data,label):
    clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                               max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                               min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0, warm_start=False, class_weight=None)#entropy
    clf.fit(data,label)
    importance=clf.feature_importances_ 
    model=SelectFromModel(clf,prefit=True)
    new_data = model.transform(data)
    return new_data,importance
