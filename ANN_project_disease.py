# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:09:17 2020

@author: admin
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("Trainingnew.csv")
x_train= dataset.iloc[:, :132].values
y_train= dataset.iloc[:, 132].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y_train = labelencoder_y_1.fit_transform(y_train)

onehotencoder = OneHotEncoder( categorical_features=[0])
y_train = onehotencoder.fit_transform(y_train.reshape(-1,1)).toarray()


test = pd.read_csv("Testing.csv")
x_test= test.iloc[:, :132].values
y_test= test.iloc[:, 132].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_2 = LabelEncoder()
y_test = labelencoder_y_2.fit_transform(y_test)

onehotencoder = OneHotEncoder( categorical_features=[0])
y_test = onehotencoder.fit_transform(y_test.reshape(-1,1)).toarray()

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier= Sequential()

classifier.add(Dense(units= 77, kernel_initializer ='uniform', activation='relu', input_dim=132))

classifier.add(Dense(units=77, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units = 41, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, epochs = 88)

y_pred = classifier.predict(x_test)

y_pred = (y_pred >0.5)

y_pred=y_pred.astype(np.int)

decoded_test= y_test.dot(onehotencoder.active_features_).astype(int)

decoded= y_pred.dot(onehotencoder.active_features_).astype(int)

de= labelencoder_y_2.inverse_transform(decoded_test)

result= labelencoder_y_2.inverse_transform(decoded)

result==de

