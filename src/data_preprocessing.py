import pandas as pd
from logger import logging
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def loading_data_for_preprocessing()->pd.DataFrame:
    try:
        df = pd.read_csv("original_dataset.csv")
        logging.info("data was successfully loaded for data preprocessing ")
        return df
    except Exception as e:
        logging.error(f"problem occured in data loading for preprocessing as {e}")

def preprocess(text):
    
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()

    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

def data_transformation(df: pd.DataFrame)->pd.DataFrame:
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

def split_and_save_data(preprocess_data:pd.DataFrame):
    train,test= train_test_split(
            preprocess_data,
            test_size=0.2,
            random_state=7,
        )
    logging.info("data was successfully divided into training and testing")

    train.to_csv("training_dataset.csv",index = False)
    logging.info("training dataset was saved successfully")

    test.to_csv("testing_dataset.csv",index = False)
    logging.info("testing dataset was saved successfully")


def main():
    df = loading_data_for_preprocessing()
    df = data_transformation(df)
    split_and_save_data(df)



if __name__ == "__main__":
    main()