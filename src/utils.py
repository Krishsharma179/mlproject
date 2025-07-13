import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill

def save_object(file_path,obj):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,"wb") as file_object:
            dill.dump(obj,file_object)
    
    except Exception as e:
        raise CustomException(e,sys)    

