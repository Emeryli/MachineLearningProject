# import
import os 
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException


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
        