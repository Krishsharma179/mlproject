import os
import sys
import pickle as pkl

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRFRegressor
from sklearn.neighbors import KNeighborsRegressor



from sklearn.metrics import r2_score

from src.exception import CustomException

from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object
from src.utils import evaluate_model


logging.info("All the file got imported")

@dataclass
class ModeltrainerConfig:
        trained_model_file_path=os.path.join("artifact","model.pkl")

class initiate_model_trainer:
        logging.info("initaited model training")
        try:
                def __init__(self):
                        self.ModelTrainerConfig=ModeltrainerConfig()
                def  initiate_model_training(self,train_arr,test_arr):
                        x_train,x_test,y_train,y_test=(
                            train_arr[:,:-1],
                            test_arr[:,:-1],
                            train_arr[:,-1], 
                            test_arr[:,-1]
                               
                        )
                        models={
                                "LinearRegression":LinearRegression(),
                                "KNeighborsRegression":KNeighborsRegressor(),
                                "AdaboostRegression":AdaBoostRegressor(),
                                "RandomForestRegression":RandomForestRegressor(),
                                "DecisionTreeRegression":DecisionTreeRegressor(),
                                "GradientBoosting":GradientBoostingRegressor(),
                                "Xgboost":XGBRFRegressor()
                        }
                        params_small = {
                                      "LinearRegression": {
                                              'fit_intercept': [True, False]
                                        },
                                        "KNeighborsRegression": {
                                             'n_neighbors': [3, 5, 7],
                                             'weights': ['uniform', 'distance']
                                             },
    
    
    
                                        "AdaboostRegression": {
                                                    'n_estimators': [50, 100, 200],
                                                  'learning_rate': [0.1, 0.5, 1.0]
                                                  },
    
                                         "RandomForestRegression": {
                                               'n_estimators': [50, 100, 200],
                                              'max_depth': [None, 10, 20],
                                              'min_samples_split': [2, 5]
                                               },
    
                                         "DecisionTreeRegression": {
                                                   'max_depth': [None, 10, 20],
                                                    'min_samples_split': [2, 5],
                                                    'min_samples_leaf': [1, 2]
                                               },
    
                                                     "GradientBoosting": {
                                                       'n_estimators': [50, 100, 200],
                                                      'learning_rate': [0.1, 0.2],
                                                        'max_depth': [3, 5, 7]
                                                },
    
                                                     "Xgboost": {
                                                          'n_estimators': [50, 100, 200],
                                                          'learning_rate': [0.1, 0.2],
                                                          'max_depth': [3, 5, 7],
                                                         'min_child_weight': [1, 3]
                                                    }
                                          }
                        model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params_small)

                        best_score=max(sorted(model_report.values()))

                        if best_score<0.6:
                            raise CustomException("NO best model")
                        
                        # getting best model name
                        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_score)]
                        # .index will give the index of best model score in model report
                        
                        best_model=models[best_model_name]

                        save_object(
                               file_path=self.ModelTrainerConfig.trained_model_file_path,
                               obj=best_model
                               
                        )
                          
                        best_model.fit(x_train,y_train)
                        y_pred=best_model.predict(x_test)

                        r2score=r2_score(y_test,y_pred)

                        return r2score

                       

                        
                
                       

                          
                

                

        except Exception as e:
                raise CustomException(e,sys)














