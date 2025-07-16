## Import necessary libraries and modules
import os
import sys
from dataclasses import dataclass
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Model Trainer Configuration
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """Model Trainer Class
    This class is responsible for training various regression models and evaluating their performance.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Initiates the model training process."""
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Support Vector Regressor': SVR(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Gaussian Naive Bayes': GaussianNB(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'XGBoost Regressor': XGBRegressor()
            }

            # Load model parameters from YAML file
            with open(os.path.join(os.path.dirname(__file__), 'model_params.yml'), 'r') as f:
                params = yaml.safe_load(f)

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                                   
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys) from e



