# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:34:21 2020

@author: Lori
"""

#%%
import sys
genpath = 'D:/Code'
if genpath not in sys.path:
    sys.path.append(genpath)

import DataSciPy
from importlib import reload
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import paper_codes_datasets.helper_knn as hlp
from scipy.sparse import vstack
import pickle
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import src.general_helper as genH
from src.nn.make_models import build_models
#%%
# Control parameters

finmod_alldat = False
fewfeat = False # drop the pubchem2d features

#%%
## =============================================================================
## Prepare data similar to Simone
final_db = pd.read_csv('paper_codes_datasets/lc_db_processed.csv')
fp = pd.DataFrame(final_db.pubchem2d.apply(DataSciPy.splt_str))
fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)

# data without pubchem2d
final_db_fewfeat  = final_db.drop(columns=['pubchem2d'])
final_db_fewfeat = final_db_fewfeat.drop(columns=['test_cas','smiles',
                                                    'fish','Unnamed: 0'])
# data with pubchem2d
final_db_manyfeat = final_db.drop(columns=['pubchem2d']).join(fp)
final_db_manyfeat = final_db_manyfeat.drop(columns=['test_cas','smiles',
                                                    'fish','Unnamed: 0'])
# Prepare binary classification problem and encode features
final_db_manyfeat = genH.binary_score(final_db_manyfeat)
final_db_fewfeat = genH.binary_score(final_db_fewfeat)

# use min max scaler (similar to Simone, and it seems reasonable looking at
# the distribution of the data)
sclr = MinMaxScaler(feature_range=(-1,1))

dummy = DataSciPy.Dataset()
dummy.setup_data(X=final_db_manyfeat.drop(columns=['score']),
                 y=final_db_manyfeat.loc[:,['score']],
                 split_test=0.33, seed=42, scaler=sclr)

dummy_few = DataSciPy.Dataset()
dummy_few.setup_data(X=final_db_fewfeat.drop(columns=['score']),
                 y=final_db_fewfeat.loc[:,['score']],
                 split_test=0.33, seed=42, scaler=sclr)

# for i in np.arange(23):
#     plt.figure()
#     plt.hist(dummy.X_train.iloc[:,i])
#     plt.title(dummy.X_train.columns[i])

encode_these = ['species','conc1_type','exposure_type','obs_duration_mean',
                'family','genus','tax_order','class',
                'application_freq_unit', 'media_type', 'control_type']
dummy_few.scale()
dummy_few.encode_categories(variables=encode_these, onehot=True)

dummy.scale()
dummy.encode_categories(variables=encode_these, onehot=True)




#%%
## =============================================================================
## Neural Networks implemented in Tensorflow
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold

if fewfeat:
    mdls = build_models(dummy_few)
else:
    mdls = build_models(dummy)

clbck = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", restore_best_weights=True, patience=100
)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# w = dummy.y_train.iloc[:,0].value_counts()
# class_weight = {0: len(dummy.y_train)/w[0]/2, 1: len(dummy.y_train)/w[1]/2}
rand_init = 3 # number of random weight inizializations
#%%
# Tensorflow does not accept boolean (or int) data, so convert to float
if fewfeat:
    X_trn = dummy_few.X_train.astype(np.float32)
    X_tst = dummy_few.X_test.astype(np.float32)
    y_trn = dummy_few.y_train.astype(np.float32)
    y_tst = dummy_few.y_test.astype(np.float32)        
else:
    X_trn = dummy.X_train.astype(np.float32)
    X_tst = dummy.X_test.astype(np.float32)
    y_trn = dummy.y_train.astype(np.float32)
    y_tst = dummy.y_test.astype(np.float32)
if finmod_alldat:
    # Combine train and test for final model
    X_trn = X_trn.append(X_tst)
    y_trn = y_trn.append(y_tst)
# Fit all models
for key in mdls:
    if finmod_alldat:
        pth_ext = 'finmod_alldat/'
    else:
        pth_ext = ''
    if fewfeat:
        pth_few = 'fewfeat/'
    else:
        pth_few = ''
    fldr = 'output/nn/simonedata/'+pth_few+pth_ext+key
    os.makedirs(fldr, exist_ok=True)
    cv = 0
    model = mdls[key]
    for trn_ind, vld_ind in kfold.split(X_trn, y_trn):
        # scale the LogP feature; it is the only one that is far from 0/1.
        # mn = np.mean(lp.iloc[trn_ind])
        # sd = np.std(lp.iloc[trn_ind])
        # X_trn.LogP = (lp - mn) / sd
        for ini in range(rand_init):
            casepath = fldr+'/cv'+str(cv)+'ini'+str(ini)
            #model.load_weights(casepath+'.h5')
            model = DataSciPy.shuffle_weights(model)
            hist = model.fit(np.array(X_trn.iloc[trn_ind,:]),
                      np.array(y_trn.iloc[trn_ind,0]),
                      validation_data=(np.array(X_trn.iloc[vld_ind,:]), np.array(y_trn.iloc[vld_ind,0])),
                      batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(casepath + '_training_hist.txt',index=False)
            model.save_weights(casepath+'.h5')
            DataSciPy.plot_history(hist, file=casepath+'_training_hist.pdf')
        cv = cv + 1
#%%