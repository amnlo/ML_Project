# -*- coding: utf-8 -*-
'''Helper functions for the KNN algorithm'''

import pandas as pd
import numpy as np
import itertools

from src.general_helper import splt_str

from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from scipy.spatial.distance import hamming
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def encode_categories(final_db, encode_these = []):
    '''Take needed features from the dataset and encode string into categorical numbers
    Inputs:
        - final_db (Pandas dataframe): cleaned dataframe
        - encode_these (list): selection of features that should be encoded. If empty (default) encode all string features
    Outputs:
        - X (Pandas dataframe): feature matrix dimension NxD, where N is the datapoints number and D the number of features
        - y (numpy array): labels array (binary or multiclass), dimension Nx1 
        - enc (sklearn OrdinalEncoder): ordinal encoder used (to be used in decoding after)
    '''
        
    # Loading data
    y = final_db.score.copy().values
    X = final_db.copy()
    X = X.drop(columns=['score'])
    
    enc = OrdinalEncoder(dtype=int)
    if len(encode_these)==0:
        encode_these = [type(X.iloc[0,er]) is str for er in range(X.shape[1])] # get types of features

    enc.fit(X.loc[:,encode_these])
    X.loc[:,encode_these] = \
        enc.transform(X.loc[:,encode_these])
        
    return X, y, enc
        
def get_features():
    '''Get list of categorical and non categorical variables for KNN algorithm (see report for more information)
    Inputs:
        - Empty
    Outputs:
        - categorical (list): list of categorical features
        - non_categorical (list): list of numerical, non categorical features'''
    
    # Define variables
    categorical = ["exposure_type", "conc1_type", "species", 'obs_duration_mean', 'class', 'tax_order', 'family', 'genus']
    
    non_categorical =[]
    
    return categorical, non_categorical


def compute_distance_matrix(X, len_X_train, cat_features = [], num_features = [], group_features={}, alpha ={}):
    '''Compute distance matrix for the KNN algorithm (see report for detail)
    Inputs:
        - X (Pandas dataframe or numpy array-like): complete feature matrix X (shape NxD)
        - len_X_train (int): length of the X_train matrix previously computed (to correctly divide the train from the test)
        - cat_features (list): list of categorical features (this is not used at the moment)
        - num_features (list): list of numerical, non categorical features (this is not used at the moment)
        - group_features (dict): grouping of features (columns) to calculate hamming's distance)
        - alpha (dict): weight for each group of features)
    Output: 
        - dist_matr_train (Pandas dataframe): distance matrix to use for trainining
        - dist_matr_test (Pandas dataframe): distance matrix to use for testing
    '''

    # Consistency check
    if len(alpha)!=len(group_features):
        raise Exception('length of dict group_features not equal to length of dict alpha')
    if len(num_features)>0:
        raise Exception('cannot deal with numerical features at the moment...')

    
    # Compute one distance matrix for each group of features
    
    for grp in group_features.keys():
        X_curr = X[group_features[grp]]
        if X_curr.shape[1]==1 and type(X_curr.iloc[0,0]) is str:# if there is only one string feature in a group
            X_curr.columns = ['col1']
            # split string and then compute hamming distance on characters
            X_curr = pd.DataFrame(X_curr.col1.apply(splt_str))
            X_curr = pd.DataFrame(X_curr.col1.tolist(), index=X_curr.index)
            dist_matr = squareform(pdist(X_curr, metric = "hamming"))
        else:
            # compute hamming distance on combination of features
            dist_matr = squareform(pdist(X_curr, metric = "hamming"))
        # multiply weighted (alpha) distance matrices of different group_features
        dist_final =+ alpha[grp] * dist_matr

    # Extract train and test matrices
    dist_matr_train = dist_matr[:len_X_train,:len_X_train]
    dist_matr_test = dist_matr[len_X_train:,:len_X_train]

    return dist_matr_train, dist_matr_test

def feature_selection_knn(X_train, X_test, y_train, y_test, categorical, non_categorical):
    '''Perform online feature selection for the KNN algorithm. Default params are used due for time constraints
    Inputs:
        - categorical (list): list of categorical features
        - non_categorical (list): list of numerical, non categorical features
        - X_train, X_test (Pandas dataframe): feature matrix splitted in train and test
        - y_train, y_test (numpy array): label for train and test
    Output: 
        - best_cat (list): list of best categorical features
        - best_non_cat (list): list of best non categorical features
    '''
 
    # Best parameter to search
    best_acc = 0
    best_cat = []
    best_non_cat = []

    # Define array of total possible features
    poss_features = np.array(categorical + non_categorical)
    
    # Do all the possible combinations of features (from 1 features to all)
    for k in range(1, 18):
        print("Starting k =",k)
        
        # All combinations with this fixed k
        poss_comb = list(itertools.combinations(range(0,17),k))
        for c in poss_comb:
            cat = []
            non_cat = []
        
        # First 12 features are categorical
        for i in list(c):
            if i in range(0, 12):
                cat.append(poss_features[i])
            else:
                non_cat.append(poss_features[i])

        # Compute distance matrix and KNN
        len_X_train = len(X_train)
        X_train_new, X_test_new = compute_distance_matrix(X_train.append(X_test), len_X_train, cat,non_cat, group_features, alpha = 1)
        
        neigh = KNeighborsClassifier(metric = 'precomputed')
        neigh.fit(X_train_new, y_train.ravel())
        y_pred = neigh.predict(X_test_new)
        
        # Find accuracy
        acc = accuracy_score(y_test, y_pred)

        # If improvement, save parameters
        if acc>best_acc:
            best_acc = acc
            best_cat = cat
            best_non_cat = non_cat
            print("Best combination found! Acc: {}, features: cat: {}, non_cat:{}".format(best_acc, best_cat, best_non_cat))
        
        # Clean variables
        del X_train_new, X_test_new
    
    return best_cat, best_non_cat


def cv_knn(X, y, cat_features = [], num_features = [], alphas = [], ks = [], leafs=[], seed=13, cv=3):
    '''Perform Cross Validation on KNN algorithm
    Inputs:
        - X (Pandas dataframe): feature matrix 
        - y (numpy array): label for X, either binary or multiclass
        - cat_features (list): list of categorical features
        - num_features (list): list of numerical, non categorical features
        - alphas (list): list of alpha to try for the distance matrix
        - ks (list): list of Ks to try for the neighbours number of the classifier
        - leafs (list): list of leaf_size to try for the classifier
        - seed (int): seed to use (fixed)
        - cv (int): number of fold to use in the CV
    Output: 
        - best_alpha (int): best alpha parameter
        - best_k (int): best K parameter
        - best_leaf (int): best leaf size parameter
    '''
     
    # Set seed and best initial params
    np.random.seed(seed)
    best_accuracy = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0
    
    # Compute distance matrix for numerical features (fixed)
    X_cat = X[cat_features]
    X_num = X[num_features]
    dist_matr_num = squareform(pdist(X_num, metric = "euclidean"))

    # Grid search on best params
    for alpha in alphas:
        
        # Compute distance matrix on categorical features (depends on alpha)
        dist_matr = alpha * squareform(pdist(X_cat, metric = "hamming"))
        dist_matr += dist_matr_num
        dist_matr = pd.DataFrame(dist_matr)
        
        for k in ks:
            for leaf in leafs:

                kf = KFold(n_splits=cv, shuffle=True)
                accs = []
                for train_index, test_index in kf.split(dist_matr):
                    
                    # Split in train and test
                    X_train = dist_matr.iloc[train_index, train_index]
                    X_test = dist_matr.iloc[test_index, train_index]
                    y_train = y[train_index]
                    y_test = y[test_index]

                    # KNN on the train, compute accuracy on test
                    neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors=k, n_jobs=-2, leaf_size=leaf)
                    neigh.fit(X_train, y_train.ravel())
                    y_pred = neigh.predict(X_test)

                    accs.append(accuracy_score(y_test, y_pred))

                # Compute average accuracy and save best parameters if actual best
                avg_acc = np.mean(accs)
                if (avg_acc > best_accuracy):
                    print("New best params found! alpha:{}, k:{}, leaf:{}, acc:{}".format(alpha, k, leaf, avg_acc))
                    best_alpha = alpha
                    best_k = k
                    best_accuracy = avg_acc
                    best_leaf = leaf

    return best_alpha, best_k, best_leaf

        
def run_knn(X_train, y_train, X_test, categorical, non_categorical, group_features, alpha, k, leaf_size):
    '''Run KNN algorithm and return predictions.
    Inputs:
        - X_train, X_test (Pandas dataframe): feature matrix splitted in train and test
        - y_train (numpy array): label for train to fit the algorithm
        - categorical (list): list of categorical features
        - non_categorical (list): list of numerical, non categorical features
        - alpha (int): alpha parameter for the distance matrix
        - k (int): K parameter for the KNN
        - leaf (int): leaf size parameter for KNN
    Output: 
        - y_pred_train (numpy array): predictions done on the train set
        - y_pred_test (numpy array): predictions done on the test set
    ''' 
    
    # Compute Distance Matrix
    len_X_train = len(X_train)
    print("Computing distance matrix ...")
    X_train_distance, X_test_distance = compute_distance_matrix(X_train.append(X_test), len_X_train, categorical, non_categorical, group_features, alpha)
    
    # Run KNN
    print("Calssifying training data ...")
    neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors=k, leaf_size=leaf_size)
    neigh.fit(X_train_distance, y_train.ravel())
    
    # Make predictions
    print("Making predictions ...")
    y_pred_train = neigh.predict(X_train_distance)
    y_pred_test = neigh.predict(X_test_distance)
    
    return y_pred_train, y_pred_test


def decode_categories(X_final_train, X_final_test, enc, encode_these):
    '''Decode categorical features from numbers to strings
    Inputs:
        - X_final_train (Pandas dataframe): final train dataset with prediction (categories as numbers)
        - X_final_test (Pandas dataframe): final train dataset with prediction (categories as numbers)
    Outputs:
        - X_final_train (Pandas dataframe): final train dataset with prediction (categories as strings)
        - X_final_test (Pandas dataframe): final train dataset with prediction (categories as strings)
    '''
         
    X_final_train.loc[:,encode_these] = \
        enc.inverse_transform(X_final_train.loc[:,encode_these])
        
    X_final_test.loc[:,encode_these] = \
        enc.inverse_transform(X_final_test.loc[:,encode_these])
        
    return X_final_train, X_final_test
    