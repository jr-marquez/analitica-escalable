from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='template')
@app.route('/',  methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        weight = request.form['weight']
        height = request.form['height']
        return redirect(url_for('result',weight=weight,height=height))
    return render_template('index.html')

@app.route('/result/<int:weight>/<int:height>', methods = ['GET','POST'])
def result(weight,height):
    # Put inputs to dataframe
    model_rf = joblib.load("cfk.pkl")
    X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
    # Get prediction
    prediction = model_rf.predict(X)[0]
    if prediction == 1:
        prediction = "male"
    else:
        prediction = "female"
    return render_template('result.html', res = prediction)

if __name__ =='__main__':
    app.run(debug=True,host="0.0.0.0", port=int("5000"))