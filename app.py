from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from logger import logging
from data_fetching import fetching_video_id, comment_fetcher
from src.model_prediction import predict
from collections import Counter
import matplotlib.pyplot as plt
app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def get_prediction(request: URLRequest):
    try:
        logging.info("Received request for prediction")
        video_id = fetching_video_id(request.url)
        comments = comment_fetcher(video_id)
        result = predict(comments)
        total_comments = len(result)
        counts = Counter(result)
        pos = counts.get("positive",0)
        neg = counts.get("negative",0)
        neu = counts.get("neutral",0)
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [pos, neu, neg]
        plt.figure()
        plt.pie(sizes, labels=labels, autopct="%1.1f%%")
        plt.title("Sentiment Distribution")
        pychart = plt.show()
        return counts, total_comments,pychart

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")