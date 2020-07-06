# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:11:59 2020

@author: PROSSENJIT
"""


import tensorflow as tf
tf.__version__
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Importing data set
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.columns
train_x = train_data.drop('label', axis = 1)
train_y = train_data['label']
train_x.columns
train_x = train_x.transpose()
train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
test_data = test_data.to_numpy()


#Making tensors (Reshaping)
img = np.reshape(train_x[9], (28,28))
plt.imshow(img)
train_x.shape()
train_x = train_x.reshape(-1, 28, 28)
test_data = test_data.reshape(-1,28,28)
#Normalizing the tensors
train_x = tf.keras.utils.normalize(train_x, axis = 1)
plt.imshow(train_x[9], cmap = plt.cm.binary)

#Creating the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, input_shape = train_x.shape, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(train_x, train_y, epochs = 10)

#Prediction
prediction = model.predict(test_data)
np.argmax(prediction[5])
plt.imshow(test_data[5])

labels = [np.argmax(prediction[i]) for i in range(len(prediction))]
labels_series = pd.Series(data = labels, index = np.arange(28000))
final = pd.DataFrame(data = labels_series, columns = 'Labels')
final.to_excel('Final.xlsx')
final.to_csv('Final.csv')














