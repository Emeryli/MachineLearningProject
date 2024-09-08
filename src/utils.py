# import
import os 
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



# save_object
def save_object(file_path, obj):
    try:
        # Get the directory
        dir_path = os.path.dirname(file_path)
        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)
        # dump the obj into file path
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

# evaluate different models    
def evaluate_models(X_train, y_train, X_test, y_test, models:dict, params):
    try:
        # create a empty list
        report = {}
        # loop all the models
        for i in range(len(list(models.keys()))):
            # get each model
            model = list(models.values())[i]
            # Get the values of each param 
            param = params[list(params.keys())[i]]
            # Grid Search and fit
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            # select the best model 
            best_model = gs.best_estimator_
            # predict the models for both train and test
            y_train_predict = best_model.predict(X_train)
            y_test_predict = best_model.predict(X_test)
            # get the r2 score
            train_model_score = r2_score(y_train, y_train_predict)
            test_model_score = r2_score(y_test, y_test_predict)
            # add the key-value (model name - score) pairs
            report[list(models.keys())[i]] = test_model_score
        # return the pairs 
        return report
        
    except Exception as e:
        raise CustomException(e,sys)
        