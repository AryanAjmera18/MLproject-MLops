import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train ,X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "LinearRegression":LinearRegression(),
                "GradientBoosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(),
                "KNN":KNeighborsRegressor(),
                "SVR":SVR()
            }
            model_report:dict =evaluate_model(X_train=X_train,y_train=y_train,models=models,y_test=y_test,X_test=X_test)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score <0.6:
                raise CustomException("No model found")
            logging.info(f"Best model found: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)
            return r2
            
            
        except Exception as e:
            raise CustomException(e,sys)
                