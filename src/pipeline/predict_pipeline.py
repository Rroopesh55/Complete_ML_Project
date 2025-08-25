import sys, os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from flask import Flask, request, jsonify

class PredictPipeline:
    def __init__(self):
        pass

class Customdata:
    def __init__( self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
        ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_data_frame(self):