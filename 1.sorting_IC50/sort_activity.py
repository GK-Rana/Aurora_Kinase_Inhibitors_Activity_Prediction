#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 01:15:14 2019

@author: rana
"""

#from matplotlib import pyplot as plt
#import numpy as np
#from scipy.cluster.hierarchy import linkage
import pandas as pd
dfile = pd.read_csv('cdk4.csv')
dfile.sort_values("IC50") #sort by column(IC50) value
dfile.to_csv('sorted.csv')