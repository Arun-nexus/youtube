from urllib.parse import urlparse, parse_qs
import os
from googleapiclient.discovery import build
import pandas as pd 
from src.model_prediction import predict

def fetching_video_id(url) -> str:
    parsed = urlparse(url)

    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        video_id = parse_qs(parsed.query).get("v")
        if video_id:
            return video_id[0]

    elif parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    raise ValueError("Invalid YouTube URL")


def comment_fetcher(video_id):
    comments = []
    api_key = os.getenv("youtube_api")

    if not api_key:
        raise ValueError("Youtube api key not found in environment variables")

    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=500
    )

    response = request.execute()

    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    data = pd.DataFrame(comments)
    data.to_csv("comment.csv",index= False)
    return data

def main(url):
    video_id = fetching_video_id(url)
    comments = comment_fetcher(video_id)
    sentiment = predict(comments)
    print(sentiment,len(sentiment))

if __name__ == "__main__":
    main("https://www.youtube.com/watch?v=7wtfhZwyrcc")