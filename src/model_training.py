import pandas as pd
from pandas import DataFrame
from lightgbm import LGBMClassifier
from data_access import load_data
import pickle
from logger import logging
from notebook.configuration_file import load_parameters

def model_practice(train:DataFrame) -> LGBMClassifier:
    try:
        logging.info("model training has been started")
        params = load_parameters()
        light_model = LGBMClassifier(max_depth=params["model_parameters"]["max_depth"],
                                     learning_rate=params["model_parameters"]["learning_rate"],
                                     n_estimators=params["model_parameters"]["n_estimators"],
                                     n_jobs= -1)
        logging.info(f"{light_model} has been called for model training")
        x,y=train.iloc[:,:-1],train.iloc[:,-1]
        light_model.fit(x,y)
        logging.info("model training has been done successfully")
        return light_model
    except Exception as e:
        logging.error(f"model training stop because {e}")
        raise

def saving_model(model:LGBMClassifier):
    try:
        logging.info("saving trained model")
        with open("model.pkl","wb") as f:
           pickle.dump(model,f)
        logging.info("model was successfully saved in model.pkl file")
    except Exception as e:
        logging.error(f"model cannot be saved because {e}")
        raise 
        
def main():
    train = load_data("training_dataset.csv")
    model = model_practice(train= train)
    saving_model(model)

if __name__ == "__main__":
    main()
