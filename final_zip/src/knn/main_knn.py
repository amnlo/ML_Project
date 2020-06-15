# -*- coding: utf-8 -*-
'''KNN algorithm implementation'''

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

from src.knn.helper_knn import *
from src.general_helper import split_dataset, build_final_datasets

def knn_algorithm(final_db, feat_sel, cross, singleclass, encode_these=[], group_features=[], seed=13):
    '''Main function to use the KNN algorithm.
    Inputs:
        - final_db (Pandas dataframe): complete db after preprocessing
        - feat_sel (boolean): flag to indicate if online feature selection must be run
        - cross (boolean): flag to indicate if cross validation must be run
        - singleclass (boolean): flag to indicate if using binary or multiclass classification
        - seed (int) (fixed): random seed
    Output: 
        - X_final_train (Pandas dataframe): final train dataset with prediction
        - X_final_test (Pandas dataframe): final train dataset with prediction
    ''' 
    # Set seed 
    np.random.seed(13)
    
    # Encoding categorical features and split
    print("Encoding categorical features for the KNN algorithm...\n")
    X, y, encoder = encode_categories(final_db, encode_these)
    print("Splitting dataset in train and test...\n")
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed)
    categorical, non_categorical = get_features()
    
    # Run feature selection if requested
    if (feat_sel):
        print("Starting feature selection for KNN algorithm. ATTENTION: THIS REQUIRES MORE OR LESS 1 HOUR.\n")
        best_cat, best_non_cat = feature_selection_knn(categorical, non_categorical, X_train, X_test, y_train, y_test)
    else:
        print("Skipping feature selection for KNN. Best features loaded.\n")
        best_cat, best_non_cat = categorical, non_categorical # Already known that using all the features gives best accuracy (see report)

    # Run cross validation if requested
    if (cross):
        print("Starting CV algorithm. ATTENTION: THIS MAY REQUIRE 9 HOURS TO COMPLETE")
        alphas = np.logspace(-3, 0, 30)
        ks = range(1, 5)
        leafs = range(10, 101, 10)
        best_alpha, best_k, best_leaf = cv_knn(X_train, y_train, best_cat, best_non_cat, alphas, ks, leafs)
    else:
        print("CV skipped. Best parameters loaded\n")
        if (singleclass):
            best_alpha = 0.010826367338740546
            best_leaf = 80
            best_k = 1
        else:
            best_alpha = 0.017433288221999882
            best_leaf = 60
            best_k = 1
    print("Best parameters for CV:\n\t-alpha: {}\n\t-k: {}\n\t-leaf_size: {}\n".format(best_alpha, best_k, best_leaf))
    
    # Run KNN algorithm
    print("Run KNN algorithm with best parameters...")
    y_pred_train, y_pred_test = run_knn(X_train, y_train, X_test, group_features, best_alpha, best_k, best_leaf)
    print("Prediction achieved!\n")
    
    # Computing accuracy
    acc = accuracy_score(y_test, y_pred_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("KNN accuracy: {}".format(acc))
    print("KNN RMSE: {}\n".format(rmse))
    
    # Build final dataset
    print("Building final dataset for future uses...\n")
    X_final_train, X_final_test = build_final_datasets(X_train, X_test, y_pred_train, y_pred_test, y_train, y_test)
    X_final_train, X_final_test = decode_categories(X_final_train, X_final_test, encoder, encode_these)
    
    
    return X_final_train, X_final_test
    
    
    