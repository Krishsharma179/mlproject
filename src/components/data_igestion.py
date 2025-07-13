# Here we try to divide the data in train test validaion etc 
import os
import sys
from src.logger import logging 
from src.exception import CustomException
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation_com import DataTransformation
from src.components.data_transformation_com import DataTranformationConfig
# In data ingestion there is going to be the input like where i am going to save the train data where we are going to save the raw data etc etc

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifact","train.csv") #this will give "artifact//train.csv"
    test_data_path:str=os.path.join("artifact","test.csv")   #this will give "artifact//test.csv"
    raw_data_path:str=os.path.join("artifact","raw.csv")     #this will give "artifact//raw.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv(r'C:\Users\krish sharma\OneDrive\Desktop\proj1\notebook\data\EDA_stud.csv')
            logging.info('Read the csv file')
            
            # Making the artifact directory os.path.dirname will give directory name "artifact"

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split started")
            # dividing the data into train set and test set
            train_set,test_set=train_test_split(df,random_state=42,test_size=0.2)
            # converting the test data and train data into csv file and storing it into the artifact folder
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys) 

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    obj2=DataTransformation()
    obj2.initiate_data_transformation(train_data,test_data)                







