import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    
    Raises:
    - CustomException: If there is an error during the saving process.
    """
    try:
        import pickle
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)