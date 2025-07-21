from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomeData
from src.pipeline.predict_pipeline import Predictpipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomeData(gender=request.form.get('gender'),
        race_ethnicity=request.form.get('race_ethnicity'),
        test_preparation_course=request.form.get('test_preparation_course'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        average=request.form.get('average'))

        pred_data=data.convert_into_dataframe()

        predict_data=Predictpipeline()
        result=predict_data.predict(pred_data)

        return render_template('home.html',result=result[0])







if __name__=="__main__":
    app.run(host='0.0.0.0')

