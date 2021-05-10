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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, mean_squared_error, confusion_matrix, f1_score, recall_score
import pandas as pd
import numpy as np
import src.general_helper as genH
from src.nn.make_models import build_models, build_hypermodel

import tensorflow as tf
from tensorflow import keras
import os
import kerastuner as kt
from kerastuner.tuners import Hyperband
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
#%%
# Control parameters

cv_fixed_models = True
logreg_sklearn = False
logreg_keras = True
fit_crafted_model = True
fit_tuned_model = True
print_best_hyper = False

final_performance = True # if true, models are fitted to the
#trainval data and evaluated on the test data

hypertuning = False # perform hyperparameter tuning?
fit_base_model = False
setup = 'onlychem'
metrics = ['accuracy']

fldr_hyper = 'D:/hypr_tox/'
fldr = 'output/nn/simonedata/'


numerical = ['atom_number', 'bonds_number','Mol', 'MorganDensity', 'LogP',
            'alone_atom_number', 'doubleBond', 'tripleBond', 'ring_number', 'oh_count', 'MeltingPoint', 'WaterSolubility']

#%%
## =============================================================================
## Prepare data similar to Simone
final_db = pd.read_csv('paper_codes_datasets/lc_db_processed.csv')
final_db = final_db.drop(columns=['Unnamed: 0'])

if setup=='onlychem':
    # remove all but chemical features and collapse duplicated rows to median conc
    tmp = numerical.copy()
    tmp.extend(['test_cas','conc1_mean'])
    final_db_cllps = final_db.loc[:,tmp]
    final_db_cllps = final_db_cllps.groupby(by = 'test_cas').agg('median').reset_index()
    final_db_cllps.index=final_db_cllps.loc[:,'test_cas']
    final_db = final_db.loc[:,['test_cas','pubchem2d']].drop_duplicates().join(final_db_cllps,on=['test_cas'],lsuffix='_lft')
    final_db = final_db.drop(columns=['test_cas_lft'])

fp = pd.DataFrame(final_db.pubchem2d.apply(DataSciPy.splt_str))
fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
fp.columns = ['pub'+ str(i) for i in range(1,882)]

# data with pubchem2d
final_db = final_db.drop(columns=['pubchem2d']).join(fp)
final_db = final_db.drop(columns=['test_cas'])

# Prepare binary classification problem
final_db = genH.binary_score(final_db)

# use min max scaler (similar to Simone, and it seems reasonable looking at
# the distribution of the data)
#sclr = MinMaxScaler(feature_range=(-1,1))
sclr = MinMaxScaler()

dummy = DataSciPy.Dataset()
dummy.setup_data(X=final_db.drop(columns=['score']),
                 y=final_db.loc[:,['score']],
                 split_test=0)

encode_these = ['species','conc1_type','exposure_type','obs_duration_mean',
                'family','genus','tax_order','class',
                'application_freq_unit', 'media_type', 'control_type']

if setup != 'onlychem':
    dummy.encode_categories(variables=encode_these, onehot=True)

# Tensorflow does not accept boolean (or int) data, so convert to float
X = dummy.X_train.astype(np.float64)
y = dummy.y_train.astype(np.float64)
    
clbck = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", restore_best_weights=False, patience=50
)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, 
                                                  test_size = 0.2, 
                                                  shuffle=True, 
                                                  random_state=42,
                                                  stratify=y)

sclr = MinMaxScaler()
sclr.fit(X_trainval[numerical])
xtmp = X_trainval.copy()
xtmp.loc[:,numerical] = sclr.transform(X_trainval.loc[:,numerical])
X_trainval = xtmp.copy()
xtmp = X_test.copy()
xtmp.loc[:,numerical]  = sclr.transform(X_test.loc[:,numerical])
X_test = xtmp.copy()
#%%
## =============================================================================
## Test LogisticRegression to have 1:1 comparison to Simone's results
if cv_fixed_models:
    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    accs = []
    sens = []
    specs = []
    
    if logreg_keras:
        mdls = build_models(dummy)
        model_logreg = mdls['model0']
    if fit_crafted_model:
        mdls = build_models(dummy)
        model_crafted = mdls['model5']
    
    cv = 0
    for train_index, val_index in kf.split(X_trainval):
        if final_performance:
            X_train = X_trainval.copy()
            X_val = X_test.copy()
            y_train = y_trainval.copy()
            y_val = y_test.copy()
            tststr = 'test/'
        else:
            X_train = X_trainval.iloc[train_index]
            X_val = X_trainval.iloc[val_index]
            y_train = y_trainval.iloc[train_index]
            y_val = y_trainval.iloc[val_index]
            tststr = ''
    
        sclr = MinMaxScaler()
        sclr.fit(X_train[numerical])
        new_train = X_train.copy()
        new_train.loc[:, numerical] = sclr.transform(X_train[numerical])
    
        new_val = X_val.copy()
        new_val.loc[:, numerical] = sclr.transform(X_val[numerical])
                
        if logreg_sklearn:
            lrc = LogisticRegression(n_jobs = -1)
            lrc.fit(new_train, y_train)
            y_pred = lrc.predict(new_val)
            y_pred_cls = np.round(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
    
            accs.append(accuracy_score(y_val, y_pred))
            sens.append(recall_score(y_val, y_pred))
            specs.append(tn/(tn+fp))
            
        if logreg_keras:
            fldr_curr = fldr+setup+'/'+'model0/'+tststr
            os.makedirs(fldr_curr, exist_ok=True)
            model_logreg = DataSciPy.shuffle_weights(model_logreg)
            hist = model_logreg.fit(new_train, y_train,
                                    validation_data=(new_val, y_val),
                                    batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(fldr_curr+'model0_training_hist_cv'+str(cv)+'.txt',index=False)
            model_logreg.save(fldr_curr+'model0_cv'+str(cv)+'.h5')
            DataSciPy.plot_history(hist, file=fldr_curr+'training_hist'+str(cv)+'.pdf')
            y_pred = model_logreg.predict(new_val)
            y_pred_cls = np.round(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
            acc = accuracy_score(y_val, y_pred_cls)
            rec = recall_score(y_val, y_pred_cls)
            spc = tn/(tn+fp)
            f1  = f1_score(y_val, y_pred_cls)
            arr = np.array([acc, rec, spc, f1])
            with open(fldr_curr+'scores.txt', 'ab') as f:
                np.savetxt(f, arr.reshape(1,4))

        
        if fit_crafted_model:
            fldr_curr = fldr+setup+'/'+'model5/'+tststr
            os.makedirs(fldr_curr, exist_ok=True)
            model_crafted = DataSciPy.shuffle_weights(model_logreg)
            hist = model_crafted.fit(new_train, y_train,
                                    validation_data=(new_val, y_val),
                                    batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(fldr+setup+'/' + 'model5/'+tststr+'model5_training_hist_cv'+str(cv)+'.txt',index=False)
            model_crafted.save(fldr_curr+'model5_cv'+str(cv)+'.h5')
            DataSciPy.plot_history(hist, file=fldr_curr+'training_hist'+str(cv)+'.pdf')
            y_pred = model_crafted.predict(new_val)
            y_pred_cls = np.round(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
            acc = accuracy_score(y_val, y_pred_cls)
            rec = recall_score(y_val, y_pred_cls)
            spc = tn/(tn+fp)
            f1  = f1_score(y_val, y_pred_cls)
            arr = np.array([acc, rec, spc, f1])
            with open(fldr_curr+'scores.txt', 'ab') as f:
                np.savetxt(f, arr.reshape(1,4))


        if fit_tuned_model:
            fldr_curr = fldr+setup+'/'+'tuned_models/'+tststr
            os.makedirs(fldr_curr, exist_ok=True)
            # load tuner
            tuner = Hyperband(
                build_hypermodel,
                objective=kt.Objective('val_accuracy', direction='max'),
                max_epochs=1000,
                factor=3,
                directory=os.path.normpath(fldr_hyper+'/'+setup+'/'),
                project_name='cv0')
            best_hps = tuner.get_best_hyperparameters(num_trials=1)
            model = tuner.hypermodel.build(best_hps[0])
            hist = model.fit(new_train, y_train,
                             validation_data=(new_val, y_val),
                             batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(fldr_curr+'model_tuned_training_hist_cv'+str(cv)+'.txt',index=False)
            model.save(fldr_curr+'model_tuned_cv'+str(cv)+'.h5')
            DataSciPy.plot_history(hist, file=fldr_curr+'model_tuned_training_hist'+str(cv)+'.pdf')
            y_pred = model.predict(new_val)
            y_pred_cls = np.round(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
            acc = accuracy_score(y_val, y_pred_cls)
            rec = recall_score(y_val, y_pred_cls)
            spc = tn/(tn+fp)
            f1  = f1_score(y_val, y_pred_cls)
            arr = np.array([acc, rec, spc, f1])
            with open(fldr_curr+'scores.txt', 'ab') as f:
                np.savetxt(f, arr.reshape(1,4))

        cv = cv + 1
        

# compare to single perceptron with keras

#%%
## =============================================================================
## Fit different Neural Networks implemented in Tensorflow with tuner

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = 0
for trn_ind, val_ind in kfold.split(np.zeros(X_trainval.shape[0]), y_trainval):
    casepath = fldr+setup+'/'+'/cv'+str(cv)+'/'
    os.makedirs(casepath, exist_ok=True)
    X_trn = X_trainval.iloc[trn_ind,:]
    X_val = X_trainval.iloc[val_ind,:]
    y_trn = y_trainval.iloc[trn_ind,:]
    y_val = y_trainval.iloc[val_ind,:]

    if fit_base_model:
        mdls = build_models(dummy)
        model = mdls['model0']

        hist = model.fit(X_trn, y_trn,
                         validation_data=(X_val, y_val),
                         batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
        pd.DataFrame.from_dict(hist.history).to_csv(casepath + 'model_base_training_hist.txt',index=False)
        model.save(casepath+'model_base.h5')
        DataSciPy.plot_history(hist, file=casepath+'model_base_training_hist.pdf')

    if hypertuning:
        clbck_hyper = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", restore_best_weights=True, patience=50
            )

        tuner = Hyperband(
            build_hypermodel,
            objective=kt.Objective('val_accuracy', direction='max'),
            max_epochs=1000,
            factor=3,
            directory=os.path.normpath(fldr_hyper+'/'+setup+'/'),
            project_name='cv'+str(cv))
        tuner.search(np.array(X_trn),
                          np.array(y_trn),
                          validation_data=(np.array(X_val), np.array(y_val)),
                          batch_size=32, epochs=1000, verbose=0, callbacks=[clbck_hyper])
    cv = cv + 1
#%%
## =============================================================================
## Look at best hyperparameters
if print_best_hyper:
    tuner = Hyperband(
        build_hypermodel,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_epochs=1000,
        factor=3,
        directory=os.path.normpath(fldr_hyper+'/'+setup+'/'),
        project_name='cv0')
    best_hps = tuner.get_best_hyperparameters(num_trials=5)
    best_hps[0].values
    best_hps[1].values
    best_hps[2].values