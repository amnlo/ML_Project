# -*- coding: utf-8 -*-
'''KNN algorithm implementation'''

import pandas as pd
import numpy as np
import DataSciPy

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from itertools import combinations

from src.knn.helper_knn import *
from src.general_helper import split_dataset, build_final_datasets, compute_tanimoto

class Knn:
    
    def __init__(self, name='dummy'):
        self.name = name
        
    def setup(self, data, group_features={}, alpha={}):
        if not type(data) is DataSciPy.Dataset:
            raise Exception('data must be of type DataSciPy.Dataset')
        self.data = data
        self.group_features = group_features
        self.alpha = alpha
    
    def condense_featuregroups(self):
        # When features are grouped, there are duplicates for each group. This function calculates the mapping of the full data set to the condensed data
        X = self.data.X_train.append(self.data.X_test)
        self.map = {}
        for grp in self.group_features.keys():
            X_f = X[self.group_features[grp]]
            # Select uniqe rows
            X_curr = X_f[~X_f.duplicated()]
            # Make a list of integers that maps from the row indices in X to the position of the condensed data
            map_condense = [-1] * len(X)
            for j in range(len(X_curr)):
                print(j, 'of', len(X_curr))
                wi = X_f.apply(lambda x,y: not (np.array(x) != np.array(y)).any(), axis=1, args=(X_curr.iloc[j,:],))
                #wi = [all(X_f.iloc[x,:] == X_curr.iloc[j,:]) for x in range(len(X_f))]
                map_condense = np.where(wi, j, map_condense)
            if any(map_condense==-1):
                raise Exception('some rows of X could not be assigned to the condensed dataset')
            self.map[grp] = map_condense

    
    def compute_distance(self, metrics={}):
        # Compute one distance matrix for each group of features
        if(len(metrics)==0): # if metrics is not provided, use 'hamming' for all groups
            metrics = self.group_features.copy()
            for grp in metrics.keys():
                metrics[grp] = 'hamming'
        X = self.data.X_train.append(self.data.X_test)
        self.distances = {}
        if len(metrics)>0:
            self.metrics = metrics
        for grp in self.group_features.keys():
            X_curr = X[self.group_features[grp]]
            # Select uniqe rows
            if hasattr(self, 'map'):
                if(len(self.map)>0):
                    X_curr = X_curr[~X_curr.duplicated()]
            # Make a list of integers that maps from the row indices in X to the position of the condensed data
            if X_curr.shape[1]==1 and type(X_curr.iloc[0,0]) is str:# if there is only one string feature in a group
                X_curr.columns = ['col1']
                # split string and then compute hamming distance on characters
                X_curr = pd.DataFrame(X_curr.col1.apply(splt_str))
                X_curr = pd.DataFrame(X_curr.col1.tolist(), index=X_curr.index)
                X_curr = X_curr=='1' # convert to boolean for better performance
            if metrics[grp]=='tanimoto':
                dist_vec = 1 - np.float32(pdist(X_curr, metric = compute_tanimoto))
            else:
                # compute the distance on combination of features
                dist_vec = np.float32(pdist(X_curr, metric = metrics[grp]))
            # save computed distance matrix in class object (because it takes so long)
            self.distances[grp] = dist_vec
            
    def construct_distance_matrix(self, alpha=0):
        if(hasattr(self, 'distances')):
            if alpha==0:
                alpha = self.alpha
            # multiply condensed weighted (alpha) distances of different group_features
            dist_weighted = self.distances.copy()
            for grp in self.group_features.keys():
                dist_weighted[grp] = np.float32(alpha[grp] * self.distances[grp])
        else:
            raise Exception('cannot weight distances since they don\'t exist')

        # Construct the full train and test distance matrices from the condensed distances of each group feature which are only for the uniq rows
        n = self.data.X_train.shape[0] + self.data.X_test.shape[0]
        dist_mat = np.zeros((n,n), dtype=np.float16)
        for grp in self.group_features.keys():
            tmp = np.float16(squareform(dist_weighted[grp]))
            if(hasattr(self, 'map')):
                # If groups were condensed, inflate the distance matrix
                tmp = tmp[self.map[grp],:]
                tmp = tmp[:,self.map[grp]]
            if tmp.shape != (n,n):
                print(tmp.shape)
                print(n,n)
                raise Exception('dimension mismatch')
            dist_mat += tmp
        self.dist_train = dist_mat[:len(self.data.X_train),:len(self.data.X_train)]
        self.dist_test = dist_mat[len(self.data.X_train):,:len(self.data.X_train)]
        
    def run(self, n_neighbors=1, leaf_size=60):
        # Run KNN
        neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors=n_neighbors, leaf_size=leaf_size)
        neigh.fit(self.dist_train, self.data.y_train.iloc[:,0].ravel())
    
        # Make predictions
        y_pred_train = neigh.predict(self.dist_train)
        y_pred_test = neigh.predict(self.dist_test)
        acc = accuracy_score(self.data.y_test, y_pred_test)
        
        return acc
    
    def fun_minim(self, x, file='dummy.txt'):
        n_neighbors = np.int(np.round(np.clip(x[0], 1, None)))
        alpha = {0:np.exp(x[1]), 1:np.exp(x[2]), 2: 1.0}
        self.construct_distance_matrix(alpha=alpha)
        acc = self.run(n_neighbors=n_neighbors)
        tmp = [n_neighbors, np.exp(x[1]), np.exp(x[2]), acc]
        if len(self.metrics)>0:
            tmp.extend(self.metrics.values())
        df = pd.DataFrame([tmp])
        df.to_csv(file, mode='a', header=False)
        
        return 1-acc
        
    

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
    
    
    