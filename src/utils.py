import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

try:
    import dill as serializer
except ImportError:  # pragma: no cover
    import pickle as serializer

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
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            serializer.dump(obj, f)
        logging.info(f"Object saved successfully at {path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple machine learning models and return their R^2 scores.
    
    Parameters:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing target.
    - models (dict): A dictionary where keys are model names and values are model instances.
    
    Returns:
    - dict: A dictionary with model names as keys and their R^2 scores as values.
    
    Raises:
    - CustomException: If there is an error during the evaluation process.
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            
            para=param[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            #model.fit(X_train, y_train) # Train the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
            
        return report
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load an object from a file using pickle.
    
    Parameters:
    - file_path (str): The path from where the object will be loaded.
    
    Returns:
    - The loaded object.
    
    Raises:
    - CustomException: If there is an error during the loading process.
    """
    try:
        with open(file_path, "rb") as f:
            obj = serializer.load(f)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)