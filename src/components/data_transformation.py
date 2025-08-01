import sys
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Data Transformation Configuration
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """Data Transformation Class
    This class is responsible for transforming the data by applying preprocessing steps
    such as handling missing values, encoding categorical variables, and scaling numerical features.
    """

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Creates a preprocessor object for data transformation."""
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender','race_ethnicity', 'parental_level_of_education','lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')), # Handling missing values for numerical features
                    ('scaler', StandardScaler()) # Scaling numerical features
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')), # Handling missing values for categorical features
                    ('onehot', OneHotEncoder(handle_unknown='ignore')), # Encoding categorical features
                    ('scaler',StandardScaler(with_mean=False)) # Scaling categorical features, avoid centering sparse matrix
                ]
            )

            logging.info("Numerical and Categorical Pipelines Created Successfully")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data Loaded Successfully for Transformation")

            preprocessor_df = self.get_data_transformer_object()

            target_column = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Input and Target Features Separated Successfully")
            logging.info("Applying Preprocessing on Training and Testing Data")

            input_feature_train_arr = preprocessor_df.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_df.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing Completed Successfully") 

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_df
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys) from e