# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:29:10 2020

@author: Lori
"""

from scipy.spatial.distance import hamming
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.preprocessing import OrdinalEncoder
import src.general_helper as genH
import pandas as pd

final_db = pd.read_csv('data/processed/final_db_update.csv')
final_db.columns

final_db = genH.binary_score(final_db)
X, y, encoder = encode_categories(final_db)
X_cat = X.loc[:,['exposure_type','species']]
dd = squareform(pdist(X_cat, metric = "hamming"))


cas_to_finger = pd.read_csv('data/processed/cas_pubchemfinger_dummy.csv', names=['cas','fingerprint'])
cas_to_finger.head()

def splt(x):
    if pd.isna(x):
        return ['NA'] * 881
    else:
        return [cr for cr in x]
a = pd.DataFrame(cas_to_finger.fingerprint.apply(splt))
b = pd.DataFrame(a.fingerprint.tolist(), index=a.index)
cas_to_finger = cas_to_finger.drop(columns=['fingerprint'])
cas_to_finger = pd.concat([cas_to_finger, b], axis=1, sort=False)
cas_to_finger.head()