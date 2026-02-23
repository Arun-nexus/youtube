import pandas as pd
from logger import logging
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data_access import load_data,save_data
from notebook.configuration_file import load_parameters
import joblib
def preprocess(text):
    try:
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = text.split()
        lemmatizer = WordNetLemmatizer()

        return " ".join([lemmatizer.lemmatize(word) for word in tokens])
    except Exception as e:
        logging.error(f"there is a problem occured while removing the phrases {e}")
        raise
def sentiment_trasformation(df:pd.DataFrame):
    try:
        logging.info("data transformation was started successfully")
        labels = {"positive": 1,
                "neutral": 0,
                "negative": -1}

        y = df.map(labels)
        y = np.array(y)
        logging.info("sentiment was labeled as -1 , 0 , 1")
        return y
    except Exception as e:
        logging.error(f" sentiment was not transformed because {e}")
        raise 

def comment_transformation(df:pd.DataFrame,params):
    try:    
        logging.info("removing punctuations from comments")
        processed_comments = df.apply(preprocess)
        logging.info("lemmatization has been started")
        logging.info("data was lemmatized successfully")
        logging.info("comments has been started vectorizing")
        vectorizer = TfidfVectorizer(max_df=0.95)
        x = vectorizer.fit_transform(processed_comments)
        x = x.toarray()
        logging.info("comments has successfully vectorized")
        joblib.dump(vectorizer,params["model"]["vectorizer"])
        return x
    except Exception as e:
        logging.error(f"problem occured in data transformation {e}")
        raise

def split_and_save_data(x:np.array,y:np.array):

    try:
        params = load_parameters()    
        train_x,train_y,test_x,test_y = train_test_split(
                x,y,
                test_size=params["train_test_split"]["test_size"],
                random_state=params["train_test_split"]["random_state"],
            )
        logging.info("data was successfully divided into training and testing")
        
        training_x_path = params["train_x"]
        training_y_path = params["train_y"]
        testing_x_path = params["test_x"]
        testing_y_path = params["test_y"]
        save_data(training_x_path,train_x)
        save_data(training_y_path,train_y)
        logging.info("training dataset was saved successfully")
        save_data(testing_x_path,test_x)
        save_data(testing_y_path,test_y)

        logging.info("testing dataset was saved successfully")
    except Exception as e:
        logging.error(f"problem occured during train test split {e}")
        raise


def main():
    params = load_parameters()
    df = load_data(params["original_dataset_path"])
    y = sentiment_trasformation(df["Sentiment"])
    x = comment_transformation(df["Comment"],params)
    split_and_save_data(x,y)



if __name__ == "__main__":
    main()