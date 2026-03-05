import boto3
import joblib
import pandas as pd
from notebook.configuration_file import load_parameters
from src.model_evaluation import model_performance
from logger import logging


def compare_and_deploy():

    try:
        logging.info("comparing new model with production model")

        s3 = boto3.client("s3")
        params = load_parameters()

        # download production model
        s3.download_file(
            params["bucket_name"],
            params["model_key"],
            params["production_path"]
        )

        production_model = joblib.load(params["production_path"])
        current_model = joblib.load(params["model"]["model_file"])

        logging.info("both models loaded successfully")

        test_x = pd.read_csv(params["test_x"])
        test_y = pd.read_csv(params["test_y"])

        production_preds = production_model.predict(test_x)
        current_preds = current_model.predict(test_x)

        logging.info("model prediction has been generated")

        production_acc = model_performance(production_preds, test_y, params)
        current_acc = model_performance(current_preds, test_y, params)

        logging.info("both models accuracy calculated")

        if current_acc > production_acc:

            logging.info(
                f"new model better by {current_acc - production_acc}"
            )

            s3.upload_file(
                params["model"]["model_file"],
                params["bucket_name"],
                params["model_key"]
            )

            logging.info("new model uploaded to S3")

        else:
            logging.info(
                f"production model still better by {production_acc - current_acc}"
            )

    except Exception as e:
        logging.error(f"model deployment failed: {e}")
        raise


if __name__ == "__main__":
    compare_and_deploy()