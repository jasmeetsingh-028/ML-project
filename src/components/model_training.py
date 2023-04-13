#try to train as many models as you can for the given problem

import os
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

##config file for path

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    
    def initiate_model_trainer(self, train_arr,test_arr):

        try:
            X_train,y_train, X_test,y_test = (
                #train_array[row, column]
                train_arr[:,:-1],  #all rows take out last cloumn
                train_arr[:,-1],  #all rows from last column
                test_arr[:,:-1], #all rows take out last cloumn
                test_arr[:,-1]  #all rows from last column
            )

            #creating the dictionary of models

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            #print(model_report)

            #best model score from the dictionary

            best_model_score = max(sorted(model_report.values()))

            #best model

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model Present")
            
            logging.info(f"best model found for the data")

            save_object(file_path= self.model_trainer_config.trained_model_file_path,
                        obj = best_model
                        )
            
            predicted = best_model.predict(X_test)

            r2_scr = r2_score(y_test, predicted)

            return r2_scr

        except Exception as e:
            raise CustomException