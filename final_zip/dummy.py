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


final_db = genH.binary_score(final_db)
X, y, encoder = encode_categories(final_db)
X_cat = X.loc[:,['exposure_type','species']]
dd = squareform(pdist(X_cat, metric = "hamming"))
