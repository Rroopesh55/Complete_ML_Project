import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass # Using @dataclass to define a configuration class for data ingestion, this is a decorator that automatically generates special methods like __init__() and __repr__()
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df= pd.read_csv('C:\\Users\\dell\\Documents\\Python\\ML_project\\notebook\\data\\stud.csv')
            logging.info("Reading the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # Create the directory if it does not exist, 'exist_ok=True' allows the directory to be created only if it does not already exist
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header = True)# Save the raw data to a CSV file without the index and with headers
            logging.info("Raw data is saved")
            
            logging.info("Initiating train test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)# Split the dataset into training and testing sets with a test size of 20% and a random state for reproducibility
            logging.info("Train test split initiated successfully")
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header = True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Exited the data ingestion method or component")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()  # This will trigger the data ingestion process when the script is run directly
    

    
        
    