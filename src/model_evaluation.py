import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
from logger import logging
from pandas import DataFrame
from matplotlib import pyplot as plt 
from data_access import load_data

def model_performance(test:DataFrame):
        
    try:
        logging.info("evaluating model performance")
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
        logging.info("model has been loaded successfully")
        x,y=test.iloc[:,:-1],test.iloc[:,-1]
        prediction = model.predict(x)
        model_accuracy = accuracy_score(y,prediction)
        model_precision = precision_score(y,prediction)
        model_recall = recall_score(y,prediction)
        model_f1 = f1_score(y,prediction)
        model_loss = mean_squared_error(y,prediction)
        logging.info("model performance has been evaluated")
        categories = ["accuracy_score","precision_score","recall_score","f1_score","mean_squared_error"]
        values = [model_accuracy,model_precision,model_recall,model_f1,model_loss]
        plt.bar(categories, values)
        plt.xlabel("Categories")
        plt.ylabel("Values")
        plt.title("model performance score")
        plt.savefig("performance.png") 
        logging.info("model performance has been saved to performance.png file")

    except Exception as e:
        logging.error(f"problem occurred during model performance as {e}")
        raise

def main():
    test = load_data("testing_dataset.csv")
    model_performance(test=test)

if __name__ == "__main__":
    main()

