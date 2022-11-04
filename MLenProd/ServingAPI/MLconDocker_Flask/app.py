from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
import joblib
from my_model.predict import make_prediction


app = Flask(__name__, template_folder='template')
@app.route('/',  methods = ['GET','POST'])
def home():
    if request.method == 'POST':  
        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        age = request.form.get("age")
        country = request.form.get("country")
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight,age,country]], columns = ["Height", "Weight","Age","Country"])
        # Get prediction
        prediction = make_prediction(X)
    else:
        prediction = ""
    if prediction == 1:
        prediction = "male"
    else:
        prediction = "female"
    return render_template("website.html", output = prediction)


if __name__ =='__main__':
    app.run(debug=True,host="0.0.0.0", port=int("5000"))