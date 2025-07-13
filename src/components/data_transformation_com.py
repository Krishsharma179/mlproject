from src.exception import CustomException
from src.logger    import logging
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.utils import save_object


logging.info("process started")
@dataclass
class DataTranformationConfig:
     preprocessor_file_path=os.path.join("artifact","preprocessor.pkl")
logging.info("DataTransformation")
class DataTransformation:
      def __init__(self):
            self.data_transformation_config=DataTranformationConfig()
      def data_transforamtion(self):
            logging.info("data transformation started")
            """

            This function is useful for data tranformaton

            """
            cat_features=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']      
            num_features=['average']
            try:
                  num_pipeline=Pipeline(
                     steps=[
                        ("impute",SimpleImputer(strategy='median')),
                        ("StandardScaler",StandardScaler())
                     ]
                  )
                  logging.info("numerical pipeline got created")
                  cat_pipeline=Pipeline(
                     steps=[
                        ("Impute",SimpleImputer(strategy='most_frequent')),
                        ("Ohe",OneHotEncoder())
                       ]
                   )
                  logging.info("Catgorical pipeline got created")
                  
                  preprocessor=ColumnTransformer(
                  [
                        ("num_pipeline",num_pipeline,num_features),
                        ("cat_pipeline",cat_pipeline,cat_features)
                  ]
                  )
                  
                  
                  return preprocessor
                  
            except Exception as e:
                  raise CustomException(e,sys)
            
        
      def initiate_data_transformation(self,train_path,test_path):
            logging.info("data transformation has started")
            try:
                  train_df=pd.read_csv(train_path)
                  test_df=pd.read_csv(test_path)

                  preprocessor_obj=self.data_transforamtion()

                  target='score'
                  numerical_feature=['score','average']

                  df_independent_train=train_df.drop(target,axis=1)
                  df_dependent_train=train_df[target]

                  df_independent_test=test_df.drop(target,axis=1)
                  df_dependent_test=test_df[target]

                  logging.info("Applying preprocessing object  on the train and test data")

                  df_independent_train_arr=preprocessor_obj.fit_transform(df_independent_train)
                  df_independent_test_arr=preprocessor_obj.transform(df_independent_test)

                  train_arr=np.c_[df_independent_train_arr,np.array(df_dependent_train)]
                  test_arr=np.c_[(df_independent_test_arr,np.array(df_dependent_test))]

                  save_object(
                  file_path=self.data_transformation_config.preprocessor_file_path,
                  obj=preprocessor_obj
                   )

                  return(
                  train_arr,test_arr,
                  self.data_transformation_config.preprocessor_file_path
                  
                   )

            
            except Exception as e:
                  raise CustomException(e,sys)
            
                  




            

             
                  

                  

            
            



