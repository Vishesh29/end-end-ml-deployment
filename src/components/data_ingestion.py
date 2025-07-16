## Read the data from source
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration
    """
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """Data Ingestion Class
    This class is responsible for ingesting data from a source, splitting it
    into training and testing datasets, and saving them to specified paths.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Started")
        try:
            # Read the data from the source
            df = pd.read_csv('end-mlprojects/end_mlprojects.egg-info/notebook/data/stud.csv')
            logging.info("Data Read Successfully")

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw Data Saved Successfully")

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data Split into Train and Test Sets")

            # Save the training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and Test Data Saved Successfully")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    logging.info("Data Ingestion and Transformation Completed Successfully")


    model_trainer = ModelTrainer()
    best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")