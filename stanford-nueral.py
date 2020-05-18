# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:39:45 2020
@author: revan
"""

#Importing libraries
import glob
import os

import numpy as np
from numpy import array
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

#For text cleaning and forming a corpus of words
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer

#to measure the time taken to run the code snippet
import time

#avoid warnings
import warnings
warnings.filterwarnings('ignore')

#for metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Keras libraries
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

#CNN implementation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def get_corpus(data):
    
    corpus = []
    spell  = SpellChecker()
    for i in range(0, len(train_df)):
        review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
        review = review.lower()
        review = review.split()
        #ps = PorterStemmer()
        #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        #for i in range(0,len(review)):
            #review[i] = spell.correction(review[i]) 
        review = ' '.join(review)
        corpus.append(review)
        
    return corpus


def tokenize_and_pad(data, vocab_length):
    
    #let's tokenize the corpus and see the length of our vocabulary (total no. of different words)   
    all_words = []
    for sentence in data:
        tokenize_word = word_tokenize(sentence)
        for word in tokenize_word:
            all_words.append(word)
    
    #use 'set' to keep only the distinct words        
    unique_words = set(all_words)
    if(vocab_length):        
        #we will round off this to next nearest hundred and that will be our vocabulary length
        vocab_length = int(math.ceil(len(unique_words)/100)*100)

    #we have our vocabulary length and let's encode our words into numbers
    #this is done using 'one_hot' from keras.preprocessing.text library and what it does is that
    #it assigns a unique number to each word in our vocabulary
    embedded_sentences = [one_hot(sentence, vocab_length) for sentence in data]
    
    return embedded_sentences, vocab_length


def build_ann(x_train,y_train,x_test,y_test,vocab_length,max_words):
    
    #building the nueral network
    model = Sequential()
    model.add(Embedding(vocab_length, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #print(model.summary())         
    model.fit(x_train, y_train, batch_size=200, epochs=3,
              validation_data=(x_test, y_test), verbose=1)
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    #print('Accuracy: {}'.format(accuracy*100))
    
    return accuracy*100, model
    

def build_cnn(x_train,y_train,x_test,y_test,vocab_length,max_words):
    
    #building the nueral network
    model = Sequential()
    model.add(Embedding(vocab_length, 32, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    
    model.fit(x_train, y_train, batch_size=200, epochs=3,
              validation_data=(x_test, y_test), verbose=1)
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    #print('Accuracy: {}'.format(accuracy*100))
    
    return accuracy*100, model


def run(train_df, test_df):
    
    #building a corpus of our train and test data
    train_corpus = get_corpus(train_df)
    test_corpus  = get_corpus(test_df)
    
    train_embed, vocab_length = tokenize_and_pad(train_corpus, 0)
    test_embed, vocab_length  = tokenize_and_pad(test_corpus,vocab_length)
    
    #the embedding layer needs an input which doesn't vary everytime. Since reviews can be of
    #any length, let's pad the reviews and make all of them of same length
    '''
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(data, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))
    '''
    
    max_words = 2500
    #padded_sentences = pad_sequences(embedded_sentences, maxlen=max_words, padding='pre')
    train_pad = pad_sequences(train_embed, maxlen=max_words, padding='post')
    test_pad  = pad_sequences(test_embed, maxlen=max_words, padding='post')
    
    x_train, y_train = train_pad, train_df['Rating']
    x_test, y_test   = test_pad, test_df['Rating']
    
    score_ann, model_ann = build_ann(x_train,y_train,x_test,y_test,vocab_length,max_words)
    score_cnn, model_cnn = build_cnn(x_train,y_train,x_test,y_test,vocab_length,max_words)
    
    scores = {'ANN': score_ann,
              'CNN': score_cnn}
    
    result = pd.DataFrame(list(scores.items()), columns = ['Model', 'Accuracy'])
    #result.set_index('Model', inplace=True)
    
    return result, model_ann, model_cnn
    


if __name__ == '__main__':
    
    #Putting all the filenames in a list and reading them into a list
    #there are a total of 12500 review in each folder
    os.chdir(r'C:\Users\revan\Downloads\Git\Stanford-review\Stanford-reviews\train\pos')
    train_pos = glob.glob('*.txt')
    train_files = []
    for filename in train_pos:
        with open(filename, "r", encoding="utf8") as file:
            train_files.append(file.read())
    
    os.chdir(r'C:\Users\revan\Downloads\Git\Stanford-review\Stanford-reviews\train\neg')
    train_neg = glob.glob('*.txt')
    for filename in train_neg:
        with open(filename, "r", encoding="utf8") as file:
            train_files.append(file.read())
    
    os.chdir(r'C:\Users\revan\Downloads\Git\Stanford-review\Stanford-reviews\test\pos')
    test_pos = glob.glob('*.txt')
    test_files = []
    for filename in test_pos:
        with open(filename, "r", encoding="utf8") as file:
            test_files.append(file.read())
        
    os.chdir(r'C:\Users\revan\Downloads\Git\Stanford-review\Stanford-reviews\test\neg')
    test_neg = glob.glob('*.txt')
    for filename in test_neg:
        with open(filename, "r", encoding="utf8") as file:
            test_files.append(file.read())
        
    #Resetting the path to our script location
    os.chdir(r'C:\Users\revan\Downloads\Git\Stanford-review')
    
    #from above, we can see that the first 12500 reviews are positive and next 12500 reviews 
    #are negative and this is the case with both our train and test sets 
    pos = list(np.ones(12500))
    neg = list(np.zeros(12500))
    rating = [*pos, *neg] #concanating the two lists with 'pos' at start 
    
    train_df = pd.DataFrame({'Review':train_files, 'Rating':rating})
    test_df  = pd.DataFrame({'Review':test_files, 'Rating':rating})
    
    #the above data has all postivew reviews at first and all negative reviews later
    #let's shuffle the rows to mix them all
    train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=0).reset_index(drop=True)
    
    result, model_ann, model_cnn = run(train_df, test_df)
    
    print(result)

     