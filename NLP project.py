# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:58:12 2020

@author: PROSSENJIT
"""


import tensorflow as tf
from tensorflow import keras
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import re
data = pd.read_csv('train.csv', usecols = ['text', 'target'])
test_data = pd.read_csv('test.csv', usecols = ['text'])



text = data['text']
text_test = test_data['text']


target = data['target']

corpus = []
for i in range(len(text)):
    comment = text[i].lower()
    comment = comment.split()
    ps = PorterStemmer()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    corpus.append(comment)
corpus_test = []
for i in range(len(text_test)):
    comment = text[i].lower()
    comment = comment.split()
    ps = PorterStemmer()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    corpus_test.append(comment)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

target = target.tolist()
train_x, test_x = corpus[0: 6090], corpus[6090:]
train_y, test_y = target[0:6090], target[6090:]
train_y = train_y.to_numpy()
tokenizer = Tokenizer(oov_token = '<OOV>')
tokenizer.fit_on_texts(train_x)
word_index = tokenizer.word_index

sequence = tokenizer.texts_to_sequences(train_x)
max_seq = max(len(seq) for seq in sequence)
padded_seq = pad_sequences(sequence, padding = 'pre', maxlen = max_seq)
vocab = len(word_index)+1

sequence_test = tokenizer.texts_to_sequences(test_x)
pad_seq_test = pad_sequences(sequence_test, padding = 'pre', maxlen = max_seq)

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim = vocab, output_dim = 16, input_length = max_seq),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units = 24, activation = 'relu'),
    tf.keras.layers.Dense(units = 24, activation = 'relu'),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )
history = model.fit(padded_seq, train_y, epochs = 25, verbose = 1)

#Visualizing
print(history)
import matplotlib.pyplot as plt
def plotting_accuracy(history, string):
    plt.plot(np.arange(1,26), history.history[string], color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Curve of accuracy')
    plt.show()
plotting_accuracy(history, 'accuracy')

from sklearn.metrics import confusion_matrix
prediction = model.predict(pad_seq_test)
for i in range(len(prediction)):
    if prediction[i]>= 0.50:
        prediction[i] = 1
    else:
        prediction[i] = 0
cm = confusion_matrix(test_y, prediction)
sum(cm)
def accuracy(cm):
    return (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])
def Predict_update(prediction, threshold):
    for i in range(len(prediction)):
        if prediction[i] >= threshold:
            prediction[i] = 1
        else:
            prediction[i] = 0
a = Predict_update(prediction, 0)
accuracy_lst = []
def cutoff_plot(test_set, target):
    threshold_lst = np.arange(0,1,0.01)
    for i in range (len(threshold_lst)):
        prediction = model.predict(test_set)
        Predict_update(prediction, threshold_lst[i])
        cm = confusion_matrix(target, prediction)
        accuracy_lst.append(accuracy(cm))
    plt.plot(threshold_lst, accuracy_lst)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Cutoff point')
    plt.show()
cutoff_plot(pad_seq_test, test_y)    

    
    
type(history)
#prediction
sequence_test = tokenizer.texts_to_sequences(test_set)
pad_seq_test = pad_sequences(sequence_test, padding = 'pre', maxlen = max_seq)
prediction = model.predict(pad_seq_test)
for i in range(len(prediction)):
    if prediction[i]>= 0.50:
        prediction[i] = 1
    else:
        prediction[i] = 0
#Predicting on the test data set
sequence_test = tokenizer.texts_to_sequences(corpus_test)
pad_seq_test = pad_sequences(sequence_test, padding = 'pre', maxlen = max_seq)
prediction = model.predict(pad_seq_test)   
for i in range(len(prediction)):
    if prediction[i]>= 0.50:
        prediction[i] = 1
    else:
        prediction[i] = 0
    
    
    
    
    
    
    
    
    
    
    
    
    

