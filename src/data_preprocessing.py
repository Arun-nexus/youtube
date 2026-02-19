import pandas as pd
from logger import logging
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from data_access import load_data
from notebook.configuration_file import load_parameters

def preprocess(text):
    try:
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = text.split()
        lemmatizer = WordNetLemmatizer()

        return " ".join([lemmatizer.lemmatize(word) for word in tokens])
    except Exception as e:
        logging.error(f"there is a problem occured while removing the phrases {e}")
        raise

def data_transformation(df: pd.DataFrame)->pd.DataFrame:
    try:    
        logging.info("data transformation was started successfully")

        sentiment = df["Sentiment"]
        comment = df["Comment"]

        labels = {"positive": 1,
                "neutral": 0,
                "negative": -1}

        y = sentiment.map(labels)
        logging.info("sentiment was labeled as -1 , 0 , 1")
        logging.info("removing punctuations from comments")
        processed_comments = comment.apply(preprocess)
        logging.info("lemmatization has been started")

        logging.info("data was lemmatized successfully")
        logging.info("comments has been started vectorizing")
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(processed_comments)
        x = x.toarray()
        logging.info("comments has successfully vectorized")
        preprocess_data = pd.DataFrame(x,y)

        return preprocess_data
    except Exception as e:
        logging.error(f"problem occured in data transformation {e}")
        raise

def split_and_save_data(preprocess_data:pd.DataFrame):

    try:
        params = load_parameters()    
        train,test= train_test_split(
                preprocess_data,
                test_size=params["train_test_split"]["test_size"],
                random_state=params["train_test_split"]["random_state"],
            )
        logging.info("data was successfully divided into training and testing")
        
        training_path = params["training_dataset"]
        testing_path = params["testing_dataset"]
        train.to_csv(training_path,index = False)
        logging.info("training dataset was saved successfully")

        test.to_csv(testing_path,index = False)
        logging.info("testing dataset was saved successfully")
    except Exception as e:
        logging.error(f"problem occured during train test split {e}")
        raise


def main():
    df = load_data("original_dataset.csv")
    df = data_transformation(df)
    split_and_save_data(df)



if __name__ == "__main__":
    main()