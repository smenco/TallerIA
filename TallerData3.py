# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:49:52 2022

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

url = 'weatherAUS.csv'

#--------------------------DATASET weatherAUS----------------------------------

data = pd.read_csv(url)

#Remplazos
rangos = [-8.0,0,10,20,30,40]
nombres = ['1','2','3','4','5']
data.MinTemp = pd.cut(data.MinTemp, rangos, labels=nombres)


rangos2 = [0,10,20,30,40,50]
nombres2 = ['1','2','3','4','5']
data.MaxTemp = pd.cut(data.MaxTemp, rangos2, labels=nombres2)


rangos3 = [-1,50,100,150,200,250]
nombres3 = ['1','2','3','4','5']
data.Rainfall = pd.cut(data.Rainfall, rangos3, labels=nombres3)


rangos4 = [-1,20,40,60,80,100]
nombres4 = ['1','2','3','4','5']
data.Evaporation = pd.cut(data.Evaporation, rangos4, labels=nombres4)


rangos5 = [-1,5,10,15]
nombres5 = ['1','2','3']
data.Sunshine = pd.cut(data.Sunshine, rangos5, labels=nombres5)



rangos6 = [0,30,60,90,130]
nombres6 = ['1','2','3','4']
data.WindGustSpeed = pd.cut(data.WindGustSpeed, rangos6, labels=nombres6)


rangos7 = [0,20,40,60,80]
nombres7 = ['1','2','3','4']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos7, labels=nombres7)


rangos8 = [0,20,40,60,80]
nombres8 = ['1','2','3','4']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos8, labels=nombres8)


rangos9 = [-1,20,40,60,80,101]
nombres9 = ['1','2','3','4','5']
data.Humidity9am = pd.cut(data.Humidity9am, rangos9, labels=nombres9)


rangos10 = [-1,20,40,60,80,101]
nombres10 = ['1','2','3','4','5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos10, labels=nombres10)


rangos11 = [980,1000,1050]
nombres11 = ['1','2']
data.Pressure9am = pd.cut(data.Pressure9am, rangos11, labels=nombres11)


rangos12 = [970,1000,1040]
nombres12 = ['1','2']
data.Pressure3pm = pd.cut(data.Pressure3pm, rangos12, labels=nombres12)


rangos13 = [-0.5,0,20,40]
nombres13 = ['1','2','3']
data.Temp9am = pd.cut(data.Temp9am, rangos13, labels=nombres13)


rangos14 = [0,20,40,50]
nombres14 = ['1','2','3']
data.Temp3pm = pd.cut(data.Temp3pm, rangos14, labels=nombres14)


data['RainToday'].replace(['No', 'Yes'], [0, 1], inplace=True)
data['RainTomorrow'].replace(['No', 'Yes'], [0, 1], inplace=True)


data.dropna(axis=0,how='any', inplace=True)


#---------------------------------------------------------------------------------------------------
#Columnas Innecesarias
data.drop(['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RISK_MM'], axis= 1, inplace = True)


#---------------------------------------------------------------------------------------------------
# partir la data en dos

data_train = data[:40000]
data_test = data[40000:]


x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)


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
print('RANDOM FOREST')

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