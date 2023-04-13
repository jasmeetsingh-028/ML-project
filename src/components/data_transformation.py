import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DatatransformationConfig():
    preprocessor_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()

    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_cols = ['writing score', 'reading score']
            categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline = Pipeline(
                steps = [           #imputer for handeling missing values and std scaler to sclae the values
                ("imputer", SimpleImputer(strategy="median")),
                ("std scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), 
                ("one_hot_endcoder", OneHotEncoder()),
                ('std_scaler', StandardScaler(with_mean= False))
                ]

            )

            logging.info(F"Column Preprocessing Pipeline created: {categorical_cols},{numerical_cols}")

            preprocessor = ColumnTransformer(
                 [
                (
                "num_pipeline",
                numerical_pipeline,
                numerical_cols
                ),
                (
                "cat_pipeline",
                categorical_pipeline,
                categorical_cols
                )

                 ]
            )

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Data reading complete')

            preprocessing_obj = self.get_data_transformer_object()

            target_col_name = 'math score'

            numerical_cols = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns = [target_col_name], axis = 1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns = [target_col_name], axis = 1)
            target_feature_test_df = test_df[target_col_name]

            logging.info('Preprocessing columns')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Preprocessing completed')

            save_object(

                file_path = self.data_transformation_config.preprocessor_object_file_path,
                obj = preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path

            )
        
        except Exception as e:
            raise CustomException(sys, e)





        