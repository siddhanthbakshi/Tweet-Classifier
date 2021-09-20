#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:27:26 2021

@author: sid
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import re
import preprocessor as p
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout



## Reading the dataset
df = pd.read_csv('Practical/train1.csv')
print(df.head())

## Check if any null values are present or not
print(df.isnull().sum())

X = df.drop('label', axis = 1)
y = df['label']
vocab_size= 5000


## Now we have to preprocess the tweets 

tweets = X.copy()

corpus = []

## using Lemmatization to solve the problem of sparse matrix 
from nltk.stem import WordNetLemmatizer

word_net = WordNetLemmatizer()

for i in range(0,len(tweets)):
    up_tweet = re.sub('^a-zA-Z0-9', " ", tweets['tweet'][i] )
    ##print(p.clean(tweets['tweet'][i]))
    up_tweet = up_tweet.lower()
    
    up_tweet = up_tweet.split()
    
    #lemmatization
    up_tweet = [word_net.lemmatize(word) for word in up_tweet if not word in stopwords.words('english')]
    
    up_tweet = ' '.join(up_tweet)
    corpus.append(up_tweet)
    
    

    
    
    
    
## Using word embedding now to vectorize the words

o_repr = [one_hot(words,vocab_size) for words in corpus]
    
print(o_repr)

## Since the max size of one value of o_repr <20    
new_length = 20
embedded_docs = pad_sequences(o_repr, padding = 'pre', maxlen = new_length)

   
    ## We will add dropout layer in between to prevent overfitting 
embedding_vector_features=40
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=new_length))
model.add(Dropout(0.2))
model.add(LSTM(100) )
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 

print(model.summary())   

X_fin = np.array(embedded_docs)
y_fin = np.array(y)



    
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X_fin, y_fin, test_size= 0.30, random_state=42)

##tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=64)


y_pred=model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix



print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)






