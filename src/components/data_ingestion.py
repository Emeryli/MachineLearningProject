# Import
import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import DataTrainer
import os

# Data Ingestion Config Class
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
      
    def initiate_data_ingestion(self):   
        logging.info("Entered the data ingestion method or component")
        try:
            # Read CSV
            df = pd.read_csv(os.path.join('notebook','data','StudentsPerformance.csv'))
            logging.info("Read dataset as dataframe")
            # Create directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            # Saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # Split the data
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            logging.info("Train test split initiated")
            # Save train and test set 
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion is completed")
            return(
                # data paths
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            CustomException(e, sys)

# Test
if __name__=="__main__":
    # Create instance of DataIngestion class
    obj = DataIngestion()
    # Call the method, assign the returned values into train and test
    train_data, test_data = obj.initiate_data_ingestion()
    # create an instance of data transformation class
    data_transformation=DataTransformation()
    # test the transformation initial transformation method
    train_arr, test_arr, _ =data_transformation.initial_transformation(train_data, test_data)
    model_trainer = DataTrainer()
    print(model_trainer.initial_model_trainer(train_arr, test_arr))
    
    