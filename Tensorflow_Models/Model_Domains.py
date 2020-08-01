# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 02:39:32 2020

@author: adity
"""
#%% Importing the libraries
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import utils

#%% Read CSV File
df = pd.read_csv('Dataset/Domains_Dataset2.csv')
df = df[pd.notnull(df['Domain'])]
'''
print(df.head(10))
print(df['Event'].apply(lambda x: len(x.split(' '))).sum()) #No. of words(text)
'''

#%% Splitting the dataset into training and test sets.
train_size = int(len(df) * .7)
train_posts = df['Event'][:train_size]
train_tags = df['Domain'][:train_size]

test_posts = df['Event'][train_size:]
test_tags = df['Domain'][train_size:]

#%% Tokenize - converts sentence(Event String) into words with indices.
max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

#%% Train text to vectorize, with one hot encoding.
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

#%% Model
batch_size = 32
epochs = 4

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

#%%
from tensorflow.keras.models import model_from_json
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    model.save_weights("model_domain.h5")
    print("Saved model")
    
#%%
json_file = open('model.json')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_domain.h5")
print("Loaded from disk")

#%%
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%% Saving the model
model.save('Models/model_domain.h5')
 
#%%
config = model.to_json()
loaded_model = tf.keras.models.model_from_json(config)

#%% Loading the model 
from tensorflow.keras.models import load_model
new_model = load_model('models/model_domain.h5')

#%%
new_model.summary()
print(new_model.optimizer) ## Adam optimizer

#%% Evaluate with test set

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


#%% For evaluating 10 events with actual and predicted domain

text_labels = encoder.classes_ 

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_posts.iloc[i][:50], "...")
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label + "\n")


#%% Predicting custom data

text_labels = encoder.classes_ 
trial = ['AWS Certification is now free']
trial = tokenize.texts_to_matrix(trial)

print('x_test shape:', x_test.shape)
print('input_test shape:', trial.shape)
prediction = model.predict(np.array([trial[0]]))
new_prediction = text_labels[np.argmax(prediction)]
print("Predicted label: " + new_prediction + "\n")

#%%
text_labels = encoder.classes_ 

import csv

with open('sample.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        print(line)
        text = [str(line)]
        text = tokenize.texts_to_matrix(text)
        prediction = model.predict(np.array([text]))
        predicted_label = text_labels[np.argmax(prediction)]
        print("Predicted label: " + predicted_label + "\n")  