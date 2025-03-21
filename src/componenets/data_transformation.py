import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object

from src.exception import CustomException
@dataclass
class DataTransformationConfig:
    preproccesor_obj_file_path =os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''This method is used to get the data transformation object'''
        try:
            numerical_columns=["writing_score","reading_score"]
            cataegorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            
            cat_pipleline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')), ## handle missing values,
                    ('onehot',OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Cat and num pipeline created")
            
            preproceesor = ColumnTransformer([
                ("num_pipleline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipleline,cataegorical_columns)
            ])
            return preproceesor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        '''This method is used to initiate the data transformation'''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessor object")
            preprocessor_obj=self.get_data_transformer_obj()
            
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"] 
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Data split completed")
            input_feature_train_array =preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array =preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            
            logging.info(f"Saved Preprocessing object")
            
            save_object(
                file_path =self.transformation_config.preproccesor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preproccesor_obj_file_path,
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)
               
            
    