# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 00:03:01 2020

@author: Prossenjit Chanda
"""


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
tweets = pd.read_csv('train.csv', usecols = ['text', 'selected_text', 'sentiment'])
test_set = pd.read_csv('test.csv', usecols = ['text', 'sentiment'])
for i in range(len(tweets)):
    if tweets['sentiment'][i] == 'positive':
        tweets['sentiment'][i] = 1
    elif tweets['sentiment'][i] == 'neutral':
        tweets['sentiment'][i] = 2
    else:
        tweets['sentiment'][i] = 3
for i in range(len(test_set)):
    if test_set['sentiment'][i] == 'positive':
        test_set['sentiment'][i] = 1
    elif test_set['sentiment'][i] == 'neutral':
        test_set['sentiment'][i] = 2
    else:
        test_set['sentiment'][i] = 3
tweets.dropna(axis = 0, inplace = True)
tweets.isna().sum()
tweets.drop_duplicates(inplace = True)
tweets.index = np.arange(0,27480)

corpus_train = []
for i in range(len(tweets)):
    tweet = re.sub('[^a-zA-Z]', ' ', tweets['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus_train.append(tweet)
    
corpus_phrase = []
for i in range(len(tweets)):
    tweet = re.sub('[^a-zA-Z]', ' ', tweets['selected_text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus_phrase.append(tweet) 
    
corpus_test = []
for i in range(len(test_set)):
    tweet = re.sub('[^a-zA-Z]', ' ', test_set['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus_test.append(tweet)    

maxlength_tweet = max(len(corpus_train[i].split()) for i in range(len(corpus_train)))
maxlength_phrase = max(len(corpus_phrase[i].split()) for i in range(len(corpus_train)))


tokenizer = Tokenizer(oov_token = '<OOV>')
tokenizer.fit_on_texts(corpus_train)
word_index = tokenizer.word_index
#type(word_index)
sequence_tweets = tokenizer.texts_to_sequences(corpus_train)
padded_sequence_tweets = pad_sequences(sequence_tweets, padding = 'post', maxlen = maxlength_tweet)
sequence_phrase = tokenizer.texts_to_sequences(corpus_phrase)
sequence_test = tokenizer.texts_to_sequences(corpus_test)

positive_words = []
neutral_words = []
negative_words = []
for i in range(len(tweets)):
    if tweets['sentiment'][i] == 1:
        positive_words.append(sequence_phrase[i])
    elif tweets['sentiment'][i] == 2:
        neutral_words.append(sequence_phrase[i])
    else:
        negative_words.append(sequence_phrase[i])

from itertools import chain

positive_list = []
i = 0
while i< len(positive_words):
    positive_list = list(chain(positive_list, positive_words[i]))
    i += 1
i = 0
neutral_list = []
while i< len(neutral_words):
    neutral_list = list(chain(neutral_list, neutral_words[i]))
    i += 1
i = 0
negative_list = []
while i< len(negative_words):
    negative_list = list(chain(negative_list, negative_words[i]))
    i += 1    
    
positive_dict_list = []  
neutral_dict_list = []
negative_dict_list = []

for num in positive_list:
    if not num in neutral_list and negative_list:
        positive_dict_list.append(num)
        
    
for num in neutral_list:
    if not num in positive_list and negative_list:
        neutral_dict_list.append(num)
        

for num in negative_list:
    if not num in positive_list and neutral_list:
        negative_dict_list.append(num)
        



test_set['selected_word'] = pd.Series([]*len(test_set), dtype = object)

for i in range(len(test_set)):
    test_set['selected_word'][i] = []
    

def model(test_set, sequence_test, positive_words, neutral_words, negative_words):
    for i in range(len(sequence_test)):
        if test_set['sentiment'][i] == 1:
            for items1 in sequence_test[i]:
                if items1 in positive_dict_list:
                    for word, item in word_index.items():
                        if item == items1:
                            test_set['selected_word'][i].append(word)
            
                    
         
        elif test_set['sentiment'][i] == 2:
            for items2 in sequence_test[i] :
                if items2 in neutral_dict_list: 
                    for word, item in word_index.items():
                        if item == items2:
                            test_set['selected_word'][i].append(word)
            
            
            
        elif test_set['sentiment'][i] == 3:
            for items3 in sequence_test[i] :
                if items3 in negative_dict_list:
                    for word, item in word_index.items():
                        if item == items3:
                            test_set['selected_word'][i].append(word)
           

                   
            
model(test_set, sequence_test, positive_list, neutral_list, negative_list)


test_set.to_csv('final.csv')

