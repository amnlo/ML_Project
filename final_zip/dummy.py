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
import random

final_db = pd.read_csv('data/processed/final_db_update.csv')

# Divide the features into classes on which Hammings distance should be computed
encode_these = ['species','conc1_type','exposure_type','class','tax_order',
                'family','genus','obs_duration_mean']

group_features = {0:['species','class','tax_order','family','genus'],
                  1:['conc1_type','exposure_type','obs_duration_mean'],
                  2:['fingerprint']}

metrics = {0:'hamming', 1:'hamming', 2:'kulsinski'}

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
    
#knnone.compute_distance(metrics=metrics)

# Optimize hyperparameters
mets = [{0:'hamming', 1:'hamming', 2:'hamming'},
        {0:'hamming', 1:'hamming', 2:'tanimoto'},
        {0:'hamming', 1:'hamming', 2:'rogerstanimoto'},
        {0:'hamming', 1:'hamming', 2:'russellrao'}]
for met in mets:
    knnone.compute_distance(metrics=met)
    for strt in range(10):
        k = random.sample([1,2,3,4,5], 1)[0]
        x0 = np.array([k,random.uniform(-3,1),random.uniform(-3,1)])
        opt.minimize(knnone.fun_minim,
                            x0=x0,
                            args=('output/knn/optim_history_mult4.csv'),
                            method='Nelder-Mead',
                            #bounds=[(1,10), (0,1), (0,1), (0,1)],
                            options={'maxfev':1000})