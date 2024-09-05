# Import
import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Config class
@dataclass
class DataTransformationConfig:
    # Path for pickle file
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

# Transformation class
class DataTransformation:
    # Initialize 
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    # Transformation pipelines
    def get_data_transformation_object(self):
        try:
            # Numerical columns
            numerical_columns= ['reading score', 'writing score']
            # Categorical columns
            categorical_columns = ['gender', 
                                   'race/ethnicity', 
                                   'parental level of education', 
                                   'lunch', 'test preparation course']
            # Num pipeline and logging info
            num_pipeline = Pipeline([
                # Handle missing values
                ('imputer', SimpleImputer(strategy='median')),
                # Scale the values
                ('scaler', StandardScaler())
            ])
            logging.info("Defined numerical columns pipeline")
            # Cat Pipeline and logging info
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
            ])
            logging.info("Defined categorical columns pipeline")
            # Transform numerical and categorical columns
            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            logging.info("Transformed numerical and categorical columns")
            # Return preprocessor
            return preprocessor
        
        except Exception as e:
            CustomException(e, sys)
    
    # class for using the datasets
    def initial_transformation(self, train_path, test_path):
        try:
            # read the data path and log the message
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test csv")
            # create an instance of get data transformation 
            preprocessor_obj = self.get_data_transformation_object()
            
            target_column_name = "math score"
            numerical_columns = ['reading score', 'writing score']
            # train_input_df
            train_input_df=train_df.drop(columns=[target_column_name],axis=1)
            # train_target_df
            train_target_df = train_df[target_column_name]
            # test_input_df
            test_input_df=test_df.drop(columns=[target_column_name], axis=1)
            # test_target_df
            test_target_df = test_df[target_column_name]
            # log the info
            logging.info("defined train & test's input & target")
            # use fit_transform on train input
            train_input_arr=preprocessor_obj.fit_transform(train_input_df)
            # use transform on test input
            test_input_arr = preprocessor_obj.transform(test_input_df)
            # Combine input and target into single array
            train_arr = np.c_[train_input_arr, np.array(train_target_df)]
            test_arr = np.c_[test_input_arr, np.array(test_target_df)]
            logging.info("Finished initialized transformation")
            # Save the object into pickle file 
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
