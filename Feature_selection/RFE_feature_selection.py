


from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def SVM_RFE_selection(k,X,y):
    svc = SVC(kernel="linear")
    svm_index = RFE(svc,k).fit(X,y.ravel()).get_support(indices=True)
    svmresult = X[:,svm_index]
    return svmresult

def LOG_RFE_selection(k,X,y):
    log = LogisticRegression(solver='liblinear')
    logresult = RFE(log,k).fit(X,y.ravel()).get_support(indices=True)
    logresult = X[:,logresult]
#    logresult.to_csv("LOG-RFE_out.csv")
    return logresult

def XGB_selection(k,X,y):
    xgb = XGBClassifier()
    xgbresult = RFE(xgb,k).fit(X,y.ravel()).get_support(indices=True)
    xgbresult = X[:,xgbresult]
#    xgbresult.to_csv("XGBoost_out.csv")
    return xgbresult