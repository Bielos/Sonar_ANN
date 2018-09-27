# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:18:31 2018

@author: DANIEL MARTINEZ BIELOSTOTZKY
"""
# Import data handling libraries
import pandas as pd
import numpy as np

# fix random seed for reproducibility
seed = 1001
np.random.seed(seed)

# Read dataset
df = pd.read_csv('sonar.csv', header=0)
X = df.iloc[:, 0:60].values
y = df.iloc[:, 60].values

# Encode target feature to integer
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset 80% for training vs 20% for test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Create ANN (60 imput units, 60 hidden units, 1 output unit) with weights following normal distribution
from keras.models import Sequential # Stochastic
from keras.layers import Dense # output = activation(dot(input, kernel) + bias)
binary_classifier = Sequential()
binary_classifier.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='sigmoid'))
binary_classifier.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# Compile using adaptive momentum optimizer with binary cross entropy as cost funtion
binary_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
binary_classifier.fit(X_train, y_train, batch_size=5, epochs = 100)

# Create and transform probabilistic predictions
y_pred = binary_classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Compute accuracy
acc = (cm[0,0] + cm[1,1]) / len(y_test)
print('Accuracy: {0}'.format(acc))