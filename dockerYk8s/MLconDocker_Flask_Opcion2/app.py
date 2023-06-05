from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__, template_folder='template')
@app.route('/',  methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        # Unpickle classifier
        clf = joblib.load("cfk.pkl")
        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        # Get prediction
        prediction = clf.predict(X)[0]
    else:
        prediction = ""
    if prediction == 1:
        prediction = "male"
    else:
        prediction = "female"
    return render_template("website.html", output = prediction)


if __name__ =='__main__':
    app.run(debug=True,host="0.0.0.0", port=int("5000"))