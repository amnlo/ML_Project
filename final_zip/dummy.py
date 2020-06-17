# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:29:10 2020

@author: Lori
"""

from src.knn.main_knn import *
import pandas as pd
import src.general_helper as genH
import scipy.optimize as opt
import pickle

final_db = pd.read_csv('data/processed/final_db_update.csv')

# Divide the features into classes on which Hammings distance should be computed
encode_these = ['species','conc1_type','exposure_type','class','tax_order',
                'family','genus','obs_duration_mean']

group_features = {0:['species','class','tax_order','family','genus'],
                  1:['conc1_type','exposure_type','obs_duration_mean'],
                  2:['fingerprint']}

metrics_tanimoto = {0:'hamming', 1:'hamming', 2:'tanimoto'}

alpha = {0: 0.0052945425292895, 1: 0.005674270695405289, 2: 1.0} # best for binary

# final_db = genH.binary_score(final_db)
# knnone = Knn()
# knnone.setup(final_db, encode_these=encode_these, group_features=group_features, alpha=alpha)
# knnone.condense_featuregroups()

pth = 'output/knn/knn_class_instances/binary_cls'
# with open(pth, 'wb') as knn_file:
#   pickle.dump(knnone, knn_file)

with open(pth, 'rb') as knn_file:
    knnone = pickle.load(knn_file)
    
knnone.compute_distance(metrics=metrics_tanimoto)

# Optimize hyperparameters
opt.minimize(knnone.fun_minim,
                        x0=np.array([3,np.log(0.1),np.log(0.1)]),
                        args=('output/knn/optim_history_tanimoto_bin2.csv'),
                        method='Nelder-Mead',
                        #bounds=[(1,10), (0,1), (0,1), (0,1)],
                        options={'maxfev':1000})