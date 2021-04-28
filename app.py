# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:02:16 2021

@author: Venkat'sAIWorld
"""

from flask import Flask, request, render_template
import pickle
import numpy as np
app=Flask(__name__)

model=pickle.load(open('hrLRModel.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    
    values=[int(x) for x in request.form.values()]
    final_feats=[np.array(values)]
    predictions=model.predict(final_feats)
    output=predictions[0]
    
    return render_template('index.html', prediction_txt= 'Promoted status is : {}'.format(output))



if __name__=="__main__":
    app.run(debug=True)