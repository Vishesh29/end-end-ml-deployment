## Web app creation

import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the target variable using the trained model.
        :param features: DataFrame containing the input features.
        :return: Predicted value.
        """
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info(f"Model loaded from {model_path}")
            logging.info(f"Preprocessor loaded from {preprocessor_path}")
            
            data_scaled = preprocessor.transform(features)

            prediction = model.predict(data_scaled)
            logging.info(f"Prediction: {prediction}")

            return prediction
        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    """
    Custom data class to handle input data for prediction.
    """    
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int,
):


        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_data_frame(self):
        """
        Converts the input data into a pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom data input as DataFrame: {df}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e


        
