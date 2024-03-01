import os , sys
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.PowerGeneration.logger import logging
from src.PowerGeneration.exception import CustomException

from src.PowerGeneration.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class Modeltrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split traning and input data")
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'SVR': SVR(),
                'XGBoost': XGBRegressor(objective='reg:squarederror', seed=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'KNN' : KNeighborsRegressor()
            }

            params = {
                'Linear Regression': {},
                'Ridge Regression': {
                    'alpha': [1.0, 0.1, 0.5, 1.5],
                },
                'Lasso Regression': {
                    'alpha': [1.0, 0.1, 0.5, 1.5],
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [5, 10, 8],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [2, 3, 10]
                },
                'SVR': {
                    'kernel': ['rbf'],
                    'C': [1.0],
                    'epsilon': [0.1]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [5, 10, 8],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [1.0]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [5, 10, 8],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [1.0]
                },
                'KNN': {
                    'n_neighbors': [5, 8, 10],
                    'weights': ['uniform'],
                    'algorithm': ['auto']
                }
            }


            model_report: dict=evaluate_models(X_train, y_train,X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)                               
                    ]
        

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No Best Model Found")
            
            logging.info(f"Best model on traning and test dataset{best_model}:{best_model_score}")


            save_obj (
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
           
            predicted = best_model.predict(X_test)

            r2_scores = r2_score(y_test, predicted) 

            return r2_scores
        except Exception as e:
            raise CustomException(e, sys)
        
    
