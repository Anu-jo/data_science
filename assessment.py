# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:10:04 2023

@author: anujo
"""

import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
data = pd.read_csv("Dry_Bean_Dataset.csv")
y = data['Class']
#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['Class'], axis = 1)
# y is dependent variable and X is independent variable 
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=100, test_size=0.3) 
cor = X_train.corr()
plt.figure(figsize=(12,10))
sns.heatmap(cor, cmap=plt.cm.CMRmap_r,annot=True)
plt.show()  
def correlation(data_sample_proportional, threshold):
    col_corr = set()  
    corr_matrix = data_sample_proportional.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
              colname = corr_matrix.columns[i]
              col_corr.add(colname)
    return col_corr      
corr_features = correlation(X_train, 0.85)
print(corr_features)
mutual_info = mutual_info_classif(X_train, y_train)
print(mutual_info)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending=False))
sel_five_cols = SelectKBest(mutual_info_classif, k=4)
sel_five_cols.fit(X_train, y_train)
print(X_train.columns[sel_five_cols.get_support()])
