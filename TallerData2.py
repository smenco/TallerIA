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

rangos3 = [0,50,100,150]
nombres3 = ['1','2','3']
data.BloodPressure = pd.cut(data.BloodPressure, rangos3, labels=nombres3)

rangos4 = [0,200,500,700,900]
nombres4 = ['1','2','3','4']
data.Insulin = pd.cut(data.Insulin, rangos4, labels=nombres4)
data.Insulin.replace(0, 1, inplace=True)

rangos5 = [0,20,40,60]
nombres5 = ['1','2','3']
data.BMI = pd.cut(data.BMI, rangos5, labels=nombres5)
data.BMI.replace(0.0,0.1, inplace=True)

rangos6 = [20,40,60,90]
nombres6 = ['1','2','3']
data.Age = pd.cut(data.Age, rangos6, labels=nombres6)
data.Pregnancies.replace(np.nan, 1, inplace=True)
data.dropna(axis=0,how='any', inplace=True)


#---------------------------------------------------------------------------------------------------
#Columnas Innecesarias
data.drop(['SkinThickness','DiabetesPedigreeFunction'], axis= 1, inplace = True)

#---------------------------------------------------------------------------------------------------
# partir la data en dos

data_train = data[:450]
data_test = data[450:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) 


#REGRESION LOGISTICA ----------------------------------------------------------

#MODELO
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

#ENTRENAMIENTO
logreg.fit(x_train,y_train)

# METRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')



# MAQUINA DE SOPORTE VECTORIAL-------------------------------------------------

# MODELO
svc = SVC(gamma='auto')

# ENTRENAMIENTO
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')



# ARBOL DE DECISIÓN------------------------------------------------------------

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')



# RANDOM FOREST------------------------------------------------------------

# Seleccionar un modelo
forest = RandomForestClassifier()

# Entreno el modelo
forest.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('NEAREST NEIGHBORS')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {forest.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {forest.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {forest.score(x_test_out, y_test_out)}')


# NAIVE BAYES------------------------------------------------------------

# Seleccionar un modelo
nayve = GaussianNB()

# Entreno el modelo
nayve.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('NAYVE BAYES')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {nayve.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {nayve.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {nayve.score(x_test_out, y_test_out)}')
