#!/usr/bin/env python
# coding: utf-8
import json
import os

from joblib import dump
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
#from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

df = pd.read_csv("Tweets.csv")
new_df = df[['text','airline_sentiment']]

tokenizer = nltk.RegexpTokenizer(r"\w+")
new_df['text'] = new_df['text'].apply(lambda x: tokenizer.tokenize(x.lower()))

from nltk.corpus import stopwords
stopWords = stopwords.words('english')
new_df['text'] = new_df['text'].apply(lambda x: [item for item in x if item not in stopWords])

new_df['words'] = 1

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()

for index,entry in enumerate(new_df['text']):
    words = []
    
    for word in entry:
        pos_word = get_wordnet_pos(word)
        words.append(lemmatizer.lemmatize(word,pos_word))
    new_df.loc[index,'words'] = str(words)
	

train_x, test_x, train_y, test_y = model_selection.train_test_split(new_df['words'],new_df['airline_sentiment'],test_size=0.2)

encoder = LabelEncoder()
train_y_encoded = encoder.fit_transform(train_y)
test_y_encoded = encoder.fit_transform(test_y)

tfidf_vect = TfidfVectorizer(max_features=100000)
tfidf_vect.fit(new_df['words'])
train_x_tfidf = tfidf_vect.transform(train_x)
test_x_tfidf = tfidf_vect.transform(test_x)


params = {"random_state":25, "solver":"lbfgs", "max_iter":1000,"multi_class":"multinomial"}
lr=LogisticRegression(**params)

lr.fit(train_x_tfidf,train_y)
pred_train=lr.predict(train_x_tfidf)
pred_test=lr.predict(test_x_tfidf)

train_acc = accuracy_score(train_y,pred_train)
test_acc = accuracy_score(test_y,pred_test)

metadata = {
		"train_acc_score": train_acc,
		"test_acc_score":test_acc
}
print("Serializing model to: {}".format(MODEL_PATH))
dump(lr, MODEL_PATH)

print("Serializing metadata to: {}".format(METADATA_PATH))
with open(METADATA_PATH, 'w') as outfile:  
    json.dump(metadata, outfile)