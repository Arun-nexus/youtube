from urllib.parse import urlparse, parse_qs
import os
from googleapiclient.discovery import build
import pandas as pd 
from src.model_prediction import predict
from notebook.configuration_file import load_parameters

def fetching_video_id(url) -> str:
    parsed = urlparse(url)

    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        video_id = parse_qs(parsed.query).get("v")
        if video_id:
            return video_id[0]

    elif parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    else:
        return
    raise ValueError("Invalid YouTube URL")


def comment_fetcher(video_id):
    if video_id is None:
        return 
    comments = []
    dates = []

    api_key = os.getenv("youtube_api")
    youtube = build("youtube", "v3", developerKey=api_key)

    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(snippet["textDisplay"])
            dates.append(snippet["publishedAt"])

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    data = pd.DataFrame({
        "comment": comments,
        "published_at": dates
    })

    return data

def main(url):
    video_id = fetching_video_id(url)
    comments = comment_fetcher(video_id)
    sentiment = predict(comments)
    print(sentiment,len(sentiment))

if __name__ == "__main__":
    main("https://www.youtube.com/watch?v=7wtfhZwyrcc")