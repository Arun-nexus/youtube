import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
from logger import logging
from pandas import DataFrame
from matplotlib import pyplot as plt 
from src.data_access import load_data
from notebook.configuration_file import load_parameters
import numpy as np

def model_predict(test_x:DataFrame,params) -> np.array:
    try:
        logging.info("evaluating model performance")
        with open(params["model"]["model_file"],"rb") as f:
            model = pickle.load(f)
        logging.info("model has been loaded successfully")
        prediction = model.predict(test_x)
        return prediction
    except Exception as e:
        logging.error(f"problem occurred during model prediciton as {e}")
        raise 

def model_performance(prediction:np.array,test_y:DataFrame,params):
        
    try:
        model_accuracy = accuracy_score(test_y,prediction)
        model_precision = precision_score(test_y,prediction,average="weighted")
        model_recall = recall_score(test_y,prediction,average="weighted")
        model_f1 = f1_score(test_y,prediction,average="weighted")
        model_loss = mean_squared_error(test_y,prediction)
        logging.info("model performance has been evaluated")
        categories = ["accuracy_score","precision_score","recall_score","f1_score","mean_squared_error"]
        values = [model_accuracy,model_precision,model_recall,model_f1,model_loss]
        plt.bar(categories, values)
        plt.xlabel("Categories")
        plt.ylabel("Values")
        plt.title("model performance score")
        plt.savefig(params["model"]["metrics"]) 
        logging.info("model performance has been saved to performance.png file")

    except Exception as e:
        logging.error(f"problem occurred during model performance as {e}")
        raise

def main():
    params = load_parameters()
    test_x = load_data(params["train_y"])
    test_y = load_data(params["test_y"])
    prediction = model_predict(test_x,params=params)
    model_performance(prediction,test_y,params=params)

if __name__ == "__main__":
    main()

