from flask import Flask, request, render_template
from redis import Redis, RedisError, StrictRedis
import os
from joblib import load
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

def get_sentiment(sentences):
	# Load model
	print("Loading model from: {}".format(MODEL_PATH))
	lr = load(MODEL_PATH)
	# #############################################################################
	tokenizer = nltk.RegexpTokenizer(r"\w+")
	text = tokenizer.tokenize(sentences.lower())
	stopWords = stopwords.words('english')
	text = [w for w in text if not w in stopWords] 
	
	

	tag_map = defaultdict(lambda : wordnet.NOUN)
	tag_map['J'] = wordnet.ADJ
	tag_map['V'] = wordnet.VERB
	tag_map['R'] = wordnet.ADV



	lemmatizer = WordNetLemmatizer()
	words=[]

	for token, tag in pos_tag(text):
		words.append(lemmatizer.lemmatize(token, tag_map[tag[0]]))

	text = str(words)
	
	text = pd.Series(text)
	#tfidf_vect = pickle.load(open("tfidf_vect.pkl", 'rb'))
	tfidf_vect = joblib.load(open("tfidf_vect.pkl", 'rb'))
	#tfidf_vect = TfidfVectorizer(max_features=100000)
	#tfidf_vect.fit(text)
	sentences_tfidf = tfidf_vect.transform(text)
	

	
	
	

	# #############################################################################
	# Run inference
	print("Scoring observations...")
	y_pred = lr.predict(sentences_tfidf)
	return (y_pred)

@app.route('/')
def index():

	
		
	
	return render_template("index.html")

@app.route('/result', methods=['POST'])	
def sent():
	if request.method == 'POST':
		result = request.form
		text = result['sentences']
		sent = get_sentiment(text)
	return render_template("result.html", sentiment = sent)

if __name__ == '__main__':
	app.run(host='0.0.0.0')




