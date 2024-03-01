import sys ,  os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.PowerGeneration.utils import save_obj

from src.PowerGeneration.logger import logging
from src.PowerGeneration.exception import CustomException




@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            numerical_columns = ["temperature", "exhaust_vacuum", 
                                 "amb_pressure", "r_humidity"]
            
            num_pipeline_steps = [
                ("imputer", SimpleImputer(strategy='median')),
                ('scalar', StandardScaler()),
            ]

            num_pipeline = Pipeline(steps=num_pipeline_steps)

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )


            return preprocessor

        except Exception as e:
           raise  CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            preprocessing_obj = self.get_data_transformation_object()

            target_column = "energy_production"
            numerical_columns = ["temperature", "exhaust_vacuum", 
                                 "amb_pressure", "r_humidity"]
            
            input_features_train_df = train_df.drop(columns= [target_column], axis = 1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns= [target_column], axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessinng on traning and test df")

            input_feature_train = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test = preprocessing_obj.transform(input_features_test_df)


            train_arr = np.c_[
                input_feature_train , np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test , np.array(target_feature_test_df)
            ]

            logging.info(f'saved preprocessing obj')

            save_obj( 
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj
                )       
        

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
    
        except Exception as e:
            raise CustomException(e, sys)
        


