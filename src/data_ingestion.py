from src.data_access import Proj1Data
from notebook.configuration_file import load_parameters
from logger import logging

proj = Proj1Data()




def loading_data_from_mongodb():
    try:
        print("data loading was started")
        df = proj.export_collection_as_dataframe("PROJECT-1-DATA")

        logging.debug("data was successfully loaded from MongoDB")
        params = load_parameters()
        orignal_dataset_path = params["original_dataset_path"]
        df.to_csv(orignal_dataset_path, index=False, header=True)

        logging.info("original dataset saved successfully on your system")

    except Exception as e:
        logging.error(f"error in data loading and saving from mongodb: {e}")
        raise


