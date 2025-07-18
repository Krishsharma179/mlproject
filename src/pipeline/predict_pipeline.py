import os       
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class Predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_file_path="artifact\model.pkl"
            preprocessor_file_path="artifact\preprocessor.pkl"
            model=load_object(file_path=model_file_path)
            preprocessor=load_object(file_path=preprocessor_file_path)

            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)

            return pred
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomeData:
    def __init__(self,gender:str,race_ethnicity:str, parental_level_of_education,lunch:str,test_preparation_course:int,average:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.average=average

    def convert_into_dataframe(self):
        dataframe={
            "gender":[self.gender],
            "race/ethnicity":[self.race_ethnicity],
            "parental level of education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test preparation course":[self.test_preparation_course],
            "average":[self.average]
        }  

        df=pd.DataFrame(dataframe)

        return df 
               
