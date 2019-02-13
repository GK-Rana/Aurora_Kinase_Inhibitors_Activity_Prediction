#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:54:52 2019

@author: rana
"""

#Step 1 importing the libraries
import numpy as np
import pandas as pd

#Step 2 importing the dataset
dataset = pd.read_csv('cdk4.csv')
x = dataset.iloc[:,0:20]
y = dataset.iloc[:,20]

#Step 3 Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)

#Step 4 Feature Scaling(Standardize the data)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Step 5 Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
np.savetxt('test.csv', X_train, delimiter=',')
np.savetxt('test.csv', X_train, delimiter=',')
import matplotlib.pyplot as plt 
from sklearn import linear_model
reg = linear_model.LinearRegression() 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_)  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 

#Step 6 Fitting logistic regression
#from sklearn.linear_model import LinearRegression 
#classifier = LinearRegression(random_state =0)
#classifier.fit(X_train, Y_train)

#Step 7 Predicting test set results
#y_pred = classifier.predict(X_test)

#Step 8 Making confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

