#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
#import joblib
from flask import Flask, request, jsonify, render_template
import pickle


# In[3]:


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


# In[4]:


@flask_app.route("/")
def Home():
    return render_template("index1.html")


# In[5]:


@flask_app.route("/predict", methods = ["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    return render_template("predict.html", prediction_text = "The Pacient is {}".format(prediction))

if __name__ == "__main__":
      flask_app.run(debug=True)
#     from waitress import serve
#     serve(flask_app, host="0.0.0.0", port=8080)

