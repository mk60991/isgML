from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

import pickle
import pandas as pd
import os
import numpy as np


# Preparing the Classifier
cur_dir = os.path.dirname('__file__')
clf = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/isg.pkl'), 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
	
	return render_template('index1.html')

@app.route('/results', methods=['POST'])
def predict():
	features1 = float(request.form['intellact'])
	features2 = float(request.form['sanskar'])
	input_data = [{'intellact': features1, 'sanskar': features2}]
	data = pd.DataFrame(input_data)
	logreg = clf.predict(data)[0]
	return render_template('results1.html', res=logreg)

if __name__ == '__main__':
	app.run()