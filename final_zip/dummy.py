# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:29:10 2020

@author: Lori
"""

from src.knn.main_knn import *
import pandas as pd
import src.general_helper as genH
import scipy.optimize as opt

final_db = pd.read_csv('data/processed/final_db_update.csv')

# Divide the features into classes on which Hammings distance should be computed
encode_these = ['species','conc1_type','exposure_type','class','tax_order','family','genus']

group_features = {0:['species','class','tax_order','family','genus'],
                  1:['conc1_type','exposure_type','obs_duration_mean'],
                  2:['fingerprint']}

alpha = {0: 0.0052945425292895, 1: 0.005674270695405289, 2: 2.7033372018729} # best for binary
alpha = {0: 0.0350870763102689, 1: 0.034663138947928135, 2: 4.7544918917910} # best for multiclass
n_neighbors = 3
best_leaf = 60

#a = list(compress(lens, ['conc1_type' in jep for jep in class_feat.values()]))

feat_sel = False
cross = False
singleclass = False

final_db = genH.binary_score(final_db)
knnone = Knn()
knnone.setup(final_db, encode_these, group_features, alpha)
knnone.compute_distances()

# Optimize hyperparameters
opt.minimize(knnone.fun_minim,
                        x0=np.array([2,0.1,0.1,2.1]),
                        args=('output/knn/optim_history_binscore3.csv'),
                        method='Nelder-Mead',
                        #bounds=[(1,10), (0,1), (0,1), (0,1)],
                        options={'maxfev':3000})

tmp2 = knnone.weight_distances(alpha=alpha)
knnone.split_distance_matrix(dist_mat=squareform(tmp2))
knnone.run(n_neighbors=5)
