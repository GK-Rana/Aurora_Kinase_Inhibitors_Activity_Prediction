#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:39:46 2019

@author: rana
"""

# import numpy package for arrays and stuff 
import numpy as np  
  
# import matplotlib.pyplot for plotting our result 
#import matplotlib.pyplot as plt 
  
# import pandas for importing csv files  
import pandas as pd 

dataset = pd.read_csv('cdk4.csv')
X = dataset.iloc[:,0:21]
y = dataset.iloc[:,21]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

#from sklearn.decomposition import PCA
#pca = PCA(n_components = 10)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)


from sklearn import svm # "Support Vector Classifier" 
clf = svm.SVC(kernel='linear') 
clf.fit(X_train, y_train)
print('Test accuracy = {0}%'.format(np.round(clf.score(X_test, y_test) * 100, 2)))

"""Method 2 : Manual c, gamma values"""
#C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
#gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

#best_score = 0
#best_params = {'C': None, 'gamma': None}
#for C in C_values:
 #   for gamma in gamma_values:
  #      svc = svm.SVC(kernel='rbf',C=C, gamma=gamma)
   #     svc.fit(X_train, y_train)
    #    score = svc.score(X_test, y_test)
     #   if score > best_score:
      #      best_score = score
       #     best_params['C'] = C
        #    best_params['gamma'] = gamma
            
#print (svm.SVC.get_params())
#print (best_score*100, best_params)
