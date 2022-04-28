# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:27:09 2022

@author: smenco
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier

url = 'bank-full.csv'

data = pd.read_csv(url)


#Remplazos
data.default.replace(['no','yes'], [0,1], inplace= True)
data.job.replace(['blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'], [0,1,2,3,4,5,6,7,8,9,10,11], inplace= True)
data.marital.replace(['married','single','divorced'], [0,1,2], inplace= True)
data.education.replace(['secondary','tertiary','primary','unknown'], [0,1,2,3], inplace= True)
data.housing.replace(['no','yes'], [0,1], inplace= True)
data.loan.replace(['no','yes'], [0,1], inplace= True)
data.contact.replace(['cellular','unknown','telephone'], [0,1,2], inplace= True)
data.poutcome.replace(['unknown','failure','other','success'], [0,1,2,3], inplace= True)
#---------------------------------------------------------------------------------------------------
rangos = [18,25,40,60,100]
nombres = ['1','2','3','4',]
data.age = pd.cut(data.age, rangos, labels=nombres)

#---------------------------------------------------------------------------------------------------
#Columnas Innecesarias
data.drop(['balance', 'day', 'month', 'duration','campaign','pdays','previous'], axis= 1, inplace = True)

#Dividir Data