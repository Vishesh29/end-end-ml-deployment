## Application server for machine learning projects

from flask import Flask, request, jsonify, render_template
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    """
    Endpoint to make predictions using the trained model.
    Expects JSON input with features for prediction.
    """
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )


            pred_df = data.get_data_as_data_frame()
            logging.info(f"Prediction DataFrame: {pred_df}")
            predict_pipeline = PredictPipeline()    
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0]) # List values

    except Exception as e:
        raise CustomException(e, sys)
    

if __name__ == "__main__":
    application.run(host="0.0.0.0")