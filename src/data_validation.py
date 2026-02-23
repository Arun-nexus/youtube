import pandas as pd
from logger import logging
from notebook.configuration_file import load_parameters

def valid():
    try:
        params = load_parameters()
        logging.info("Data validation started")
        df = pd.read_csv(params["original_dataset_path"])
        logging.info("Dataset loaded for validation")

        expected_cols = {"Comment", "Sentiment"}
        assert set(df.columns) == expected_cols
        expected_labels = {"positive", "neutral", "negative"}
        assert set(df["Sentiment"].unique()) == expected_labels

        logging.info("Dataset is valid")

    except Exception as e:
        logging.error(f"Dataset validation failed: {e}")
        raise
if __name__ == "__main__":
    valid()