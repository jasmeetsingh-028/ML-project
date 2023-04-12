##data ingestion 
##big data team collects data from different sources and stores it in a databases
##you read the data from the data source

import os
import sys
#print(sys.path)
sys.path.append('C:/Users/sjasm/Documents/ML project/src')
from src import exception, logger
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
#important
from dataclasses import dataclass   #used to create class variable


@dataclass    #decorator 
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  #train.csv will be saved later at this path
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  #three paths saved in this variable

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion mathod")
        try:
            df = pd.read_csv('Notebooks and dataset\StudentsPerformance.csv')
            logging.info('Data read successful')

            #creating directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)

            #saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train test split initiated')

            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,       
                #this info will be required later for data transformations
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":

    obj = DataIngestion()
    obj.initiate_data_ingestion()

