# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv('Events_Dataset.csv')
df.head()

#%%
from io import StringIO
col = ['Type', 'Event']
df = df[col]
df = df[pd.notnull(df['Event'])]
df.columns = ['Type', 'Event']
df.head()

#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Type').Event.count().plot.bar(ylim=0)
plt.show()

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Event).toarray()
labels = df.Type
print(features.shape)

#%%
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
X_train, X_test, y_train, y_test = train_test_split(df['Event'], df['Type'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LinearSVC().fit(X_train_tfidf, y_train)


#%%
import csv

with open('try.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        print(line)
        print(clf.predict(count_vect.transform([str(line)])))

#%%
import pickle

filename = 'mt_type.sav'
pickle.dump(clf, open(filename, 'wb'))

#%%
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(count_vect.transform(["CodeBoost codeathon is live now!"]))
print(result) 


