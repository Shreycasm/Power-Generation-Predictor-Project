from src.PowerGeneration.logger import logging
from src.PowerGeneration.exception import CustomException
import sys
from src.PowerGeneration.components.data_ingestion import DataIngestion , DataIngestionConfig
from src.PowerGeneration.components.data_transformation import DataTransformation , DataTransformationConfig
from src.PowerGeneration.components.model_trainer import ModelTrainerConfig , Modeltrainer

if __name__ == "__main__":
    logging.info("run started")

    try: 
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr ,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path )

        model_trainer = Modeltrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

    except Exception as e:
        logging.info(f"error: {e}")
        raise ConnectionResetError(e, sys)