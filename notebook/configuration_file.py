import yaml
from logger import logging

def load_parameters() -> dict:
    try:
        with open("params.yaml", 'r') as file:
            params = yaml.safe_load(file)
        logging.debug(f"parameters retrieved from source params.yaml")
        return params
    except FileNotFoundError as e:
        logging.error(f"file not found on params.yaml {e}")
        raise