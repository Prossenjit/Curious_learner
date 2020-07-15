# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:54:34 2020

@author: Prossenjit Chanda
"""


import os
import zipfile

#For training set
local_zip = 'C:/Users/Prossenjit Chanda/Downloads/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/Users/Prossenjit Chanda/Desktop/Tensorflow/Rock paper ceassor')
zip_ref.close()
#For test set
local_zip_test = 'C:/Users/Prossenjit Chanda/Downloads/rps-test-set.zip'
zip_ref_test = zipfile.ZipFile(local_zip_test, 'r')
zip_ref_test.extractall('C:/Users/Prossenjit Chanda/Desktop/Tensorflow/Rock paper ceassor')
zip_ref.close()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_data_path = 'C:/Users/Prossenjit Chanda/Desktop/Tensorflow/Rock paper ceassor/rps'
test_data_path = 'C:/Users/Prossenjit Chanda/Desktop/Tensorflow/Rock paper ceassor/rps-test-set'

train_data_generator =  train_datagen.flow_from_directory(train_data_path, 
                                                          target_size = (150,150),
                                                          batch_size = 32,
                                                          class_mode = 'categorical')
test_data_generator =  test_datagen.flow_from_directory(test_data_path, 
                                                          target_size = (150,150),
                                                          batch_size = 32,
                                                          class_mode = 'categorical')
import tensorflow as tf
from tensorflow import keras
import glob
from PIL import Image
import Image
import pillow
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding= 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding= 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3),padding= 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(512, activation = 'relu' ),
    tf.keras.layers.Dense(3, activation = 'softmax')
        ])

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
history = model.fit_generator(train_data_generator, validation_data = test_data_generator, epochs = 25, verbose = 1)

history.history['val_accuracy']

import matplotlib.pyplot as plt
import numpy as np
def plotting(history, param1, param2):
    plt.plot(np.arange(25), history.history[param1], color = 'blue')
    plt.plot(np.arange(25), history.history[param2], color = 'red')
    plt.title('Accuracy plot')
    plt.show()

plotting(history, 'accuracy', 'val_accuracy')






