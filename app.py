from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from logger import logging
from data_fetching import fetching_video_id, comment_fetcher
from src.model_prediction import predict, monthly_response
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str


@app.post("/predict")
async def get_prediction(request: URLRequest):
    try:
        logging.info("Prediction request received")

        video_id = fetching_video_id(request.url)

        comments_df = comment_fetcher(video_id)

        if comments_df is None:
            raise HTTPException(status_code=400, detail = "invalid youtube url")

        predictions = predict(comments_df)

        monthly_data = monthly_response(comments_df, predictions)

        counts = Counter(predictions)

        return {
            "total_comments": len(predictions),
            "sentiment_distribution": {
                "positive": counts.get("positive", 0),
                "neutral": counts.get("neutral", 0),
                "negative": counts.get("negative", 0),
            },
            "monthly_counts": monthly_data
        }

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")