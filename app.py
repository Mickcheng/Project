from flask import Flask, request, render_template
from redis import Redis, RedisError, StrictRedis

app = Flask(__name__)

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

def get_sentiment(sentences):
	# #############################################################################
	# Load model
	print("Loading model from: {}".format(MODEL_PATH))
	clf = load(MODEL_PATH)

	# #############################################################################
	# Run inference
	print("Scoring observations...")
	y_pred = clf.predict(sentences)
	print(y_pred)



@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':	
		text = request.form['sentences']
		return get_sentiment(text)


if __name__ == '__main__':
	app.run(host='0.0.0.0')




