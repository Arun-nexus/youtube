import pandas as pd
from pandas import DataFrame
from lightgbm import LGBMClassifier
from src.data_access import load_data
import pickle
from logger import logging
import os
from notebook.configuration_file import load_parameters

def model_practice(train_x:DataFrame,train_y:DataFrame) -> LGBMClassifier:
    try:
        logging.info("model training has been started")
        params = load_parameters()
        light_model = LGBMClassifier(max_depth=params["model_parameters"]["max_depth"],
                                     learning_rate=params["model_parameters"]["learning_rate"],
                                     n_estimators=params["model_parameters"]["n_estimators"],
                                     n_jobs= -1)
        logging.info(f"{light_model} has been called for model training")

        light_model.fit(train_x,train_y)
        logging.info("model training has been done successfully")
        return light_model
    except Exception as e:
        logging.error(f"model training stop because {e}")
        raise

def saving_model(model:LGBMClassifier):
    try:
        logging.info("saving trained model")
        params = load_parameters()
        model_path = params["model"]["model_file"]
        os.makedirs(os.path.dirname(model_path),exist_ok =  True)
        with open(model_path,"wb") as f:
           pickle.dump(model,f)
        logging.info("model was successfully saved in model.pkl file")
    except Exception as e:
        logging.error(f"model cannot be saved because {e}")
        raise 
        
def main():
    params = load_parameters()
    train_x = load_data(params["train_x"])
    train_y = load_data(params["test_x"])
    model = model_practice(train_x.squeeze(),train_y.squeeze())
    saving_model(model)

if __name__ == "__main__":
    main()
