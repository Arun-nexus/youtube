from src.data_preprocessing import preprocess
from src.model_evaluation import model_predict
from logger import logging
from notebook.configuration_file import load_parameters
from src.data_access import save_data
from pandas import DataFrame
import numpy as np
import joblib
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt


def predict(comments: DataFrame)-> np.array:
    try:
        logging.info("Starting prediction")
        params = load_parameters()
        comments = comments["comment"]
        transformed_comment = comments.apply(preprocess)
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


def monthly_response(comments_df ,prediction: np.array):
    try:
        logging.info("Calculating monthly response")
        df = comments_df.copy()

        if len(prediction) != len(df):
            raise ValueError("Prediction length does not match dataframe length")

        df["response"] = prediction

        df["published_at"] = pd.to_datetime(df["published_at"])
        df["year_month"] = df["published_at"].dt.strftime("%Y-%m")

        monthly_counts = (
            df.groupby(["year_month", "response"])
            .size()
            .unstack(fill_value=0)
        )

        monthly_percent = monthly_counts.div(
            monthly_counts.sum(axis=1),
            axis=0
        ) * 100
        logging.info("Monthly response calculated successfully")
        return monthly_percent.round(2).to_dict(orient="index")


    except Exception as e:
        logging.error(f"Problem occurred during calculating monthly response: {e}")
        raise
        
    