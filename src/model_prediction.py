from src.data_preprocessing import preprocess
from src.model_evaluation import model_predict
from logger import logging
from notebook.configuration_file import load_parameters
from src.data_access import save_data
from pandas import DataFrame
import numpy as np
import joblib

def predict(comments: DataFrame):
    try:
        logging.info("Starting prediction")
        params = load_parameters()
        transformed_comment = comments[0].apply(preprocess)
        vectorizer = joblib.load(params["model"]["vectorizer"])
        vectorized_comment = vectorizer.transform(transformed_comment)
        prediction = model_predict(vectorized_comment, params)



        label_map = {
            0: "neutral",
            1: "positive",
            -1: "negative"
        }

        prediction = [label_map[no] for no in prediction]

        logging.info("Prediction successful")

        return prediction

    except Exception as e:
        logging.error(f"Prediction failed because {e}")
        raise

if __name__ == "__main__":
    prediciton = predict()
    print(len(prediciton),prediciton)
    