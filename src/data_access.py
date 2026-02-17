import sys
import pandas as pd
import numpy as np
from typing import Optional
import os
from mongodb.mongodb_connection import mongodbclient


class Proj1Data:

    def __init__(self) -> None:
        
        try:
            self.mongo_client = mongodbclient(database_name=os.getenv("database_name"))
        except Exception as e:
            raise Exception(e)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            print("Fetching data from mongoDB")
            df = pd.DataFrame(list(collection.find()))
            print(f"Data fecthed with len: {len(df)}")
            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df

        except Exception as e:
            raise Exception(e, sys)