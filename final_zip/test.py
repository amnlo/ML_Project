# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:04:19 2020

@author: Lori
"""

import sys
genpath = 'D:/Code'
if genpath not in sys.path:
    sys.path.append(genpath)
    
import DataSciPy
from src.knn.main_knn import *
import pandas as pd
import src.general_helper as genH
import scipy.optimize as opt

final_db_epfl = pd.read_csv('data/processed/final_db_processed.csv')

# Divide the features into classes on which Hammings distance should be computed
encode_these_epfl = ['ring_number', "exposure_type", "conc1_type", "species",
                           'tripleBond', 'obs_duration_mean', 'doubleBond', 'alone_atom_number',
                           'class', 'tax_order', 'family', 'genus','exposure_type', 'conc1_type',
                           'species', 'obs_duration_mean', 'class', 'tax_order', 'family', 'genus']

group_features_epfl = {0: ['ring_number', "exposure_type", "conc1_type", "species",
                           'tripleBond', 'obs_duration_mean', 'doubleBond', 'alone_atom_number',
                           'class', 'tax_order', 'family', 'genus','exposure_type', 'conc1_type',
                           'species', 'obs_duration_mean', 'class', 'tax_order', 'family', 'genus'],
                       1: ['atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']}

metrics_epfl = {0:'hamming', 1:'euclidean'}

alpha_epfl_bin = {0:  0.010826367338740546, 1: 1} # best for binary
alpha_epfl_mult = {0: 0.017433288221999882, 1: 1} # best for multiclass
n_neighbors_epfl = 1
best_leaf_epfl_bin = 80
best_leaf_epfl_mult = 60

final_db_epfl_bin  = genH.binary_score(final_db_epfl)
final_db_epfl_mult = genH.multi_score(final_db_epfl)

# Prepare binary run
dummy = DataSciPy.Dataset()
dummy.setup_data(X=final_db_epfl_bin.drop(columns=['test_cas','score']),
                 y=final_db_epfl_bin.loc[:,['score']],
                 split_test=0.3,
                 seed=13)
dummy.encode_categories(variables=encode_these_epfl)
kep = Knn()
kep.setup(dummy, group_features=group_features_epfl, alpha=alpha_epfl_bin)
kep.compute_distance(metrics=metrics_epfl)
kep.construct_distance_matrix(alpha=alpha_epfl_bin)
acc = kep.run(n_neighbors=n_neighbors_epfl, leaf_size=best_leaf_epfl_bin) # should be 0.902857

# Prepare multiclass run
dummy = DataSciPy.Dataset()
dummy.setup_data(X=final_db_epfl_mult.drop(columns=['test_cas','score']),
                 y=final_db_epfl_mult.loc[:,['score']],
                 split_test=0.3,
                 seed=13)
dummy.encode_categories(variables=encode_these_epfl)
kep = Knn()
kep.setup(dummy, group_features=group_features_epfl, alpha=alpha_epfl_mult)
kep.compute_distance(metrics=metrics_epfl)
kep.construct_distance_matrix(alpha=alpha_epfl_mult)
acc = kep.run(n_neighbors=n_neighbors_epfl, leaf_size=best_leaf_epfl_mult) # should be 0.7401428