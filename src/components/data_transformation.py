import sys
import dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/preprocessor.pkl')
    
class DataTransformation:
    def__init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function creates a preprocessor object that includes pipelines for numerical and categorical feature transformations.
        It handles missing values, scales numerical features, and encodes categorical features.
        '''
        try:
           numeric_features = ['writing_score', 'reading_score']
            categorical_features =[
                "gender",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder()),
                    ('scaler', standaredScaler())
                     
                ]
            )
            
            logging.infor("Numerical and categorical feature transformation pipelines created successfully.")
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            logging.info("Preprocessor object created successfully.")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pdf.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully.")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name ="math_score"
            numerical_features = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Input and target features separated successfully.")
            
            input_feature_train = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[
                input_feature_test, np.array(target_feature_test_df)
                ]
            logging.info("Saved Proprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        
        except Exception as e:
            raise CustomException(e, sys)

