import os
import sys

import numpy as np
import pandas as pd
import dill ## create pkl file

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save the object to the specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {str(e)}", sys) from e
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Evaluates multiple regression models and returns their performance metrics."""
    model_report = {}

    for i in range(len(list(models))):
        try:
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[list(models.keys())[i]]

            grid = GridSearchCV(model, param, cv=3)

            grid.fit(X_train, y_train) ## fit the model

            model.set_params(**grid.best_params_) ## set the best parameters
            model.fit(X_train, y_train) ## fit the model with best parameters

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            model_report[model_name] = r2
            logging.info(f"{model_name} - R2: {r2}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

            return model_report

        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")
            model_report[model_name] = -1



def load_object(file_path):
    """
    Load an object from the specified file path.
    """
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)

    except Exception as e:
        raise CustomException(e,sys)