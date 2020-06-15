# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:29:10 2020

@author: Lori
"""

import run
import pandas as pd
from src.knn.helper_knn import *
import src.general_helper as genH

final_db = pd.read_csv('data/processed/final_db_update.csv')

# Divide the features into classes on which Hammings distance should be computed
encode_these = ['species','conc1_type','exposure_type','class','tax_order','family','genus']

group_features = {0:['species','class','tax_order','family','genus'],
                  1:['conc1_type','exposure_type','obs_duration_mean'],
                  2:['fingerprint']}
alpha = {0:0.333, 1:0.333, 2:0.333}
best_leaf = 60
best_k = 1

#a = list(compress(lens, ['conc1_type' in jep for jep in class_feat.values()]))

feat_sel = False
cross = False
singleclass = False

final_db = genH.binary_score(final_db)
X, y, encoder = encode_categories(final_db, encode_these)
X_train, X_test, y_train, y_test = genH.split_dataset(X, y, seed)
categorical, non_categorical = get_features()
y_pred_train, y_pred_test = run_knn(X_train, y_train, X_test, categorical, non_categorical, group_features, alpha, best_k, best_leaf)

# Computing accuracy
acc = accuracy_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
print("KNN accuracy: {}".format(acc))
print("KNN RMSE: {}\n".format(rmse))

X_cat = X.loc[:,['exposure_type','species']]
a = pdist(X_cat, metric = "hamming")
dd = squareform(a)


cas_to_finger = pd.read_csv('data/processed/cas_pubchemfinger_dummy.csv', names=['cas','fingerprint'])
cas_to_finger.head()


a = pd.DataFrame(cas_to_finger.fingerprint.apply(splt))
b = pd.DataFrame(a.fingerprint.tolist(), index=a.index)
cas_to_finger = cas_to_finger.drop(columns=['fingerprint'])
cas_to_finger = pd.concat([cas_to_finger, b], axis=1, sort=False)
cas_to_finger.head()