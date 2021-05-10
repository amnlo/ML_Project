# -*- coding: utf-8 -*-
'''Helper functions needed for both implementations'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def binary_score(final_db):
    '''Set label to 0 or 1 based on the selected threshold of 1 mg/L
        Inputs:
            - final_db (Pandas dataframe): cleaned dataframe 
        Outputs:
            - final_db_scored (Pandas dataframe): dataframe with column score containing the labels'''
            
    # Computing score
    final_db_scored = final_db.copy()
    final_db_scored["score"] = np.where(final_db_scored.conc1_mean > 1, 0, 1)
    # Drop conc1_mean (it is replaced by score)
    final_db_scored = final_db_scored.drop(columns=['conc1_mean'])
    
    # Info
    num0 = len(final_db_scored[final_db_scored.score==0])
    num1 = len(final_db_scored[final_db_scored.score==1])
    print("There are {} datapoints with label 0 and {} datapoints with label 1".format(num0, num1))
    return final_db_scored


def scores_cat(conc):
    '''Return score of the concentration passed as input
        Inputs:
            - conc (float): concentration value
        Outputs:
            - score (int): score given to the concentration, from 1 to 5 (see report for detail)'''
    
    if (conc < 10**-1):
        return 5
    elif ((conc>=10**-1) and (conc<10**0)):
        return 4
    elif ((conc>=10**0) and (conc<10**1)):
        return 3
    elif ((conc>=10**1) and (conc<10**2)):
        return 2
    else:
        return 1


def multi_score(final_db):
    '''Set label to 0 or 1 based on the selected threshold of 1 mg/L
        Inputs:
            - final_db (Pandas dataframe): cleaned dataframe 
        Outputs:
            - final_db_scored (Pandas dataframe): dataframe with column score containing the labels'''
            
    # Computing score
    final_db_scored = final_db.copy()
    final_db_scored["score"] = final_db_scored.conc1_mean.apply(lambda x: scores_cat(x))
    # Drop conc1_mean (it is replaced by score)
    final_db_scored.drop(columns=['conc1_mean'])
    
    # Info
    for i in range(1, 6):
        num = len(final_db_scored[final_db_scored.score==i])
        print("There are {} datapoints with label {}".format(num, i))

    return final_db_scored


def split_dataset(X, y=[], seed=13):
    '''Split dataset in 30% test and 70% train
    Inputs:
        - X (Pandas dataframe or numpy array-like): feature matrix for KNN or complete matrix with score included for MF
        - y (numpy array): labels (for KNN) or empty array (for MF)
        - seed (int): seed to use (fixed)
    Outputs:
        - X_train, X_test (Pandas dataframe): feature matrix (KNN) or complete matrix (MF) splitted in train and test
        - y_train, y_test (numpy array): label for train and test (KNN)'''
    
    if (len(y)!=0):
        return train_test_split(X, y, test_size = 0.3, shuffle=True, random_state=seed)
    else:
        X_train, X_test = train_test_split(X, test_size = 0.3, shuffle=True, random_state=seed)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        return X_train, X_test
    

def build_final_datasets(X_train, X_test, y_pred_train, y_pred_test, y_train=[], y_test=[]):
    '''Build final dataset containing all the features used in the respective algorithm.
    Note: dataset built using matrix factorization will have just the experiments features
    Inputs:
        - X_train, X_test (Pandas dataframe): feature matrix for KNN or complete matrix with score for MF, splitted in train and test
        - y_pred_train, y_pred_test (numpy array): scores predictions for train and test
        - y_pred_train, y_pred_test (numpy array): original label for training (only for KNN)
    Output: 
        - X_final_train (Pandas dataframe): final train dataset with prediction
        - X_final_test (Pandas dataframe): final train dataset with prediction
    ''' 
    
    # Copy dataset for easy use
    X_final_train = X_train.copy()
    X_final_test = X_test.copy() 
    
    # Save original score
    if (len(y_train)==0 or len(y_test)==0):
        X_final_train.rename(columns={"score":"original_score"}, inplace=True)
        X_final_test.rename(columns={"score":"original_score"}, inplace=True)
    else:
        X_final_train["original_score"] = y_train
        X_final_test["original_score"] = y_test
    
    # Save prediction
    X_final_train["predicted_score"] = y_pred_train
    X_final_test["predicted_score"] = y_pred_test
    
    return X_final_train, X_final_test

def compute_tanimoto(a, b):
    a_and_b = (a & b).sum()
    not_a = ~a
    not_b = ~b
    tani = a_and_b / ((a & not_b).sum() + (b & not_a).sum() + a_and_b)

    return tani
    