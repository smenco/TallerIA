# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:03:51 2022

@author: smenco
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

url = 'diabetes.csv'

#--------------------------DATASET DIABETES------------------------------------

data = pd.read_csv(url)

#Remplazos
rangos = [0,6,12,17]
nombres = ['1','2','3']
data.Pregnancies = pd.cut(data.Pregnancies, rangos, labels=nombres)
data.Pregnancies.replace(np.nan, 1, inplace=True)
data.dropna(axis=0,how='any', inplace=True)

rangos2 = [0,50,100,150,200]
nombres2 = ['1','2','3','4']
data.Glucose = pd.cut(data.Glucose, rangos2, labels=nombres2)
data.dropna(axis=0,how='any', inplace=True)

rangos3 = [0,50,100,150]
nombres3 = ['1','2','3']
data.BloodPressure = pd.cut(data.BloodPressure, rangos3, labels=nombres3)
data.dropna(axis=0,how='any', inplace=True)

rangos4 = [0,200,500,700,900]
nombres4 = ['1','2','3','4']
data.Insulin = pd.cut(data.Insulin, rangos4, labels=nombres4)
data.dropna(axis=0,how='any', inplace=True)
data.Insulin.replace(np.nan, '1', inplace=True)


#---------------------------------------------------------------------------------------------------
#Columnas Innecesarias
data.drop(['SkinThickness', '', '', '','','',''], axis= 1, inplace = True)