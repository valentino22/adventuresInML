from flask import Flask, request, render_template, jsonify
from pycaret.classification import *
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

location = '/Users/szabi/projects/adventuresInML/kaggle/titanic/data'
fullpath = os.path.join(location, 'model_pycaret_pipeline')
model = load_model(fullpath)

cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    print(final)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen)
    print('Prediction: \n', prediction)
    survival = int(prediction.Label[0])
    score = float(prediction.Score[0])
    return render_template('home.html',
                           pred=survival,
                           likelihood="{0:.1f}%".format(score * 100))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
