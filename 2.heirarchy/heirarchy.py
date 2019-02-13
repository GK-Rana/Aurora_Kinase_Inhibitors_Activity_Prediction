#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 02:46:11 2019

@author: rana
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np 

data = pd.read_csv('cdk4.csv')
X = data.iloc[1:,1:]
Y = data.iloc[1:,21:]

#X[np.isnan(X)] = np.median(X[~np.isnan(X)])
#Y[np.isnan(Y)] = np.median(Y[~np.isnan(Y)])
np.where(X.values >= np.finfo(np.float64).max)
X

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)

