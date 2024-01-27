# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:23:26 2023

@author: anujo
"""
f = open('document.txt','r')
docs = f.read().split(".")
print(docs)
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words('english')
print(sw)
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
filtered_docs = []
for doc in docs:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in sw:
            tmp += ps.stem(w) + " "
    filtered_docs.append(tmp)
print(filtered_docs)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_docs)
print(X.todense())
from sklearn.cluster import KMeans
K = 3 
model = KMeans(n_clusters=K)
model.fit(X)
print("cluster no. of input documents, in the order they received:")
print(model.labels_)
Y = vectorizer.transform(["More snow and rain is falling in the Arctic"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Chancellor announces £1m Manchester Prize for AI"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["UK weather extremes to become new normal"])
prediction = model.predict(Y)
print(prediction)

