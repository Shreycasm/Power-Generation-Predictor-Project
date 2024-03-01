import os, sys
from src.PowerGeneration.exception import CustomException
from src.PowerGeneration.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

import pickle 
import numpy as np 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading Sql DataBase Started")
    try:
        mydb = pymysql.connect(
            host  = host,
            password = password,
            user = user,
            db = db
        )
        logging.info(f"Conection Established {mydb}")
        df = pd.read_sql_query("select *  from raw_data", mydb)
        return df
    
    except Exception as e:
        raise CustomException(e, sys)
    

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train , y_train , X_test, y_test, models, params):
        try:
            report={}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                param = params[list(models.keys())[i]]

                grid_search = GridSearchCV(model, param, cv=3)
                grid_search.fit(X_train , y_train)

                model.set_params(**grid_search.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred) 

                report[list(models.keys())[i]] = test_model_score

            return report
        
        except Exception as e:
            raise CustomException(e, sys)
    