from src.data_access import Proj1Data
import yaml
from logger import logging

proj = Proj1Data()

def load_parameters(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug(f"parameters retrieved from source {params_path}")
        return params
    except FileNotFoundError as e:
        logging.error(f"file not found on {params_path} {e}")
        raise


def load_data():
    try:
        print("data loading was started")
        df = proj.export_collection_as_dataframe("PROJECT-1-DATA")

        logging.debug("data was successfully loaded from MongoDB")

        df.to_csv("original_dataset.csv", index=False, header=True)

        logging.info("original dataset saved successfully on your system")

    except Exception as e:
        logging.error(f"error in data loading and saving from mongodb: {e}")
        raise


if __name__ == "__main__":
    load_data()
