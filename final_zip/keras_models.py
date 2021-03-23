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
#%%
# Control parameters

finmod_alldat = False

#%%
## =============================================================================
## Prepare data similar to Simone
final_db = pd.read_csv('paper_codes_datasets/lc_db_processed.csv')
fp = pd.DataFrame(final_db.pubchem2d.apply(DataSciPy.splt_str))
fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
#fp = fp.astype(int)
final_db_manyfeat = final_db.drop(columns=['pubchem2d']).join(fp)
final_db_manyfeat = final_db_manyfeat.drop(columns=['test_cas','smiles',
                                                    'fish','Unnamed: 0'])
# Prepare binary classification problem and encode features
final_db_manyfeat = genH.binary_score(final_db_manyfeat)

# use min max scaler (similar to Simone, and it seems reasonable looking at
# the distribution of the data)
sclr = MinMaxScaler()
dummy = DataSciPy.Dataset()
dummy.setup_data(X=final_db_manyfeat.drop(columns=['score']),
                 y=final_db_manyfeat.loc[:,['score']],
                 split_test=0.33, seed=42, scaler=sclr)

# for i in np.arange(23):
#     plt.figure()
#     plt.hist(dummy.X_train.iloc[:,i])
#     plt.title(dummy.X_train.columns[i])

encode_these = ['species','conc1_type','exposure_type','obs_duration_mean',
                'family','genus','tax_order','class',
                'application_freq_unit', 'media_type', 'control_type']
dummy.encode_categories(variables=encode_these, onehot=True)
dummy.scale()


#%%
## =============================================================================
## Perceptrons
## Fit perceptrons
acc_test = np.zeros((100,))
acc_train = np.zeros((len(acc_test,)))
for i in range(len(acc_test)):
    per = Perceptron(tol=1e-3, random_state=i)
    per.fit(dummy.X_train, dummy.y_train.iloc[:,0])
    acc_test[i] = per.score(dummy.X_test, dummy.y_test.iloc[:,0])
    acc_train[i] = per.score(dummy.X_train, dummy.y_train.iloc[:,0])
    
## Plot perceptron performance
plt.hist(acc_test)
plt.hist(acc_train)
print(np.max(acc_test))
print(np.max(acc_train))
#%%
## =============================================================================
## Neural Networks implemented in Tensorflow
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold

# Define models
model0 = keras.models.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1)
model0.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model0.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model0.add(keras.layers.Dropout(rate=0.2))
model0.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model0.add(keras.layers.Dropout(rate=0.2))
model0.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model0.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

model1 = keras.models.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1)
model1.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model1.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model1.add(keras.layers.Dropout(rate=0.2))
model1.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model1.add(keras.layers.Dropout(rate=0.2))
model1.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model1.add(keras.layers.Dropout(rate=0.2))
model1.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model1.add(keras.layers.Dropout(rate=0.2))
model1.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model1.add(keras.layers.Dropout(rate=0.2))
model1.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

model2 = keras.models.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1)
model2.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model2.add(keras.layers.Dense(200, activation='relu', kernel_initializer=initializer))
model2.add(keras.layers.Dropout(rate=0.2))
model2.add(keras.layers.Dense(50, activation='relu', kernel_initializer=initializer))
model2.add(keras.layers.Dropout(rate=0.2))
model2.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

model3 = keras.models.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1)
model3.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model3.add(keras.layers.Dense(1800, activation='relu', kernel_initializer=initializer))
model3.add(keras.layers.Dropout(rate=0.2))
model3.add(keras.layers.Dense(900, activation='relu', kernel_initializer=initializer))
model3.add(keras.layers.Dropout(rate=0.2))
model3.add(keras.layers.Dense(200, activation='relu', kernel_initializer=initializer))
model3.add(keras.layers.Dropout(rate=0.2))
model3.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
adm = tf.keras.optimizers.Adam(learning_rate=0.0001) # adams with smaller learning rate than default
model3.compile(optimizer=adm, loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

#model4 is similar to model2, but has tanh activation
model4 = keras.models.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=1)
model4.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model4.add(keras.layers.Dense(1800, activation='tanh', kernel_initializer=initializer))
model4.add(keras.layers.Dropout(rate=0.3))
model4.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model4.add(keras.layers.Dropout(rate=0.3))
model4.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model4.add(keras.layers.Dropout(rate=0.3))
model4.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model4.add(keras.layers.Dropout(rate=0.3))
model4.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model4.add(keras.layers.Dropout(rate=0.3))
model4.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model4.add(keras.layers.Dropout(rate=0.3))
model4.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

model5 = keras.models.Sequential()
initializer = tf.keras.initializers.he_normal(seed=1)
model5.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model5.add(keras.layers.Dense(200, activation='relu', kernel_initializer=initializer))
model5.add(keras.layers.Dropout(rate=0.3))
model5.add(keras.layers.Dense(50, activation='relu', kernel_initializer=initializer))
model5.add(keras.layers.Dropout(rate=0.3))
model5.add(keras.layers.Dense(50, activation='relu', kernel_initializer=initializer))
model5.add(keras.layers.Dropout(rate=0.3))
model5.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model5.add(keras.layers.Dropout(rate=0.3))
model5.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model5.add(keras.layers.Dropout(rate=0.3))
model5.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
model5.add(keras.layers.Dropout(rate=0.3))
model5.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

model6 = keras.models.Sequential()
initializer = tf.keras.initializers.glorot_normal(seed=1)
model6.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model6.add(keras.layers.Dense(200, activation='tanh', kernel_initializer=initializer))
model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(50, activation='tanh', kernel_initializer=initializer))
model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(50, activation='tanh', kernel_initializer=initializer))
model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
#model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
#model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
#model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
#model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
#model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
#model6.add(keras.layers.Dropout(rate=0.5))
model6.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
model6.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
model6.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
model6.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model6.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

model7 = keras.models.Sequential()
initializer = tf.keras.initializers.glorot_normal(seed=1)
model7.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model7.add(keras.layers.Dense(200, activation='tanh', kernel_initializer=initializer))
model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(50, activation='tanh', kernel_initializer=initializer))
model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(50, activation='tanh', kernel_initializer=initializer))
model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
#model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
#model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
#model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
#model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
#model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
#model7.add(keras.layers.Dropout(rate=0.5))
model7.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
model7.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
model7.add(keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
model7.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model7.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

#model8 is similar to model4
model8 = keras.models.Sequential()
initializer = tf.keras.initializers.glorot_normal(seed=1)
model8.add(keras.Input(shape=(dummy.X_train.shape[1],)))
model8.add(keras.layers.Dense(1600, activation='tanh', kernel_initializer=initializer))
model8.add(keras.layers.Dropout(rate=0.3))
model8.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model8.add(keras.layers.Dropout(rate=0.3))
model8.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model8.add(keras.layers.Dropout(rate=0.3))
model8.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model8.add(keras.layers.Dropout(rate=0.3))
model8.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model8.add(keras.layers.Dropout(rate=0.3))
model8.add(keras.layers.Dense(20, activation='tanh', kernel_initializer=initializer))
model8.add(keras.layers.Dropout(rate=0.3))
model8.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
model8.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 'cross-entropy' is the same as the 'log-loss' of scikitlearn



# Make a dict of the models to be fitted
mdls = {#'model0':model0,
        #'model1':model1,
        #'model2':model2,
        #'model3':model3
        'model7':model7,
        'model8':model8}

clbck = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", restore_best_weights=True, patience=200
)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# w = dummy.y_train.iloc[:,0].value_counts()
# class_weight = {0: len(dummy.y_train)/w[0]/2, 1: len(dummy.y_train)/w[1]/2}
rand_init = 2 # number of random weight inizializations
#%%
# Tensorflow does not accept boolean (or int) data, so convert to float
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
    fldr = 'output/nn/simonedata/'+pth_ext+key
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
                      batch_size=32, epochs=2000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(casepath + '_training_hist.txt',index=False)
            model.save_weights(casepath+'.h5')
            DataSciPy.plot_history(hist, file=casepath+'_training_hist.pdf')
        cv = cv + 1
#%%