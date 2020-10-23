##StackPDB:Predicting DNA-binding proteins based on XGB_RFE feature optimization and Stacked ensemble classifier

##StackPDB uses the following dependencies:
 * Python 3.6
 * numpy
 * scipy
 * scikit-learn
 * pandas

##Guiding principles: **The dataset contains both training dataset and independent test set.

**feature extraction:
  * EDT.py implements EDT.
  * RPT.py implements RPT.
  * mainpseaac.m and PAAC.m implement PseAAC.
  * PsePSSM.m implements PsePSSM.
  *  PSSM-TPC.py implement PSSM-TPC.
   
** feature selection:
  * LLE_SVD.py implement LLE and SVD.
  * XGB_selection.py and RFE_feature_selection.py implement elasticNet,LASSO, LinearSVC, SVM_RFE and XGB_RFE.

** Classifier:
  * 1075_stacking_knife.py implements stacked ensemble classifier.
  * GBM_1075_knife_no.py implements LightGBM.
  * XGB_1075_knife_no.py implements XGBoost.
  * GBDT_1075_knife_no.py implements GBDT.
  * KNN_1075_knife_no.py implements KNN.
  * NB_1075_knife_no.py implements NB.
  * RF_1075_knife_no.py implements RF.
  * LR_1075_knife_no.py implements LR.
  * SVM_1075_knife_no.py implements SVM.
  *  Adaboost_1075_knife_no.py implements Adaboost.
  
** Dataset:
  * Training_dataset_PDB1075.txt contains the data of the training dataset.  
  * independent_test_dataset_PDB180.txt and independent_test_dataset_PDB186.txt contains the data of the independent test set.  