import requests
import csv
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_FILE = "utils/video_links.txt"
OUTPUT_FILE = "utils/video_urls_titles.csv"

# Extract video IDs from the text file
with open(VIDEO_FILE, "r") as f:
    video_ids = f.read().splitlines()

def check_videos_content(video_ids):
    videos = []
    base_url = 'https://www.googleapis.com/youtube/v3/videos'
    batch_size = 50  # max allowed per API call

    for i in range(0, len(video_ids), batch_size):
        batch_ids = video_ids[i:i + batch_size] 
        params = {
            'part': 'snippet',
            'id': ','.join(batch_ids),
            'key': API_KEY
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        videos.extend(data['items'])
    return videos

if __name__ == "__main__":
    videos = check_videos_content(video_ids)
    
    # Write video urls and titles to CSV
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Video URL', 'Title'])
        
        for video in videos:
            vid = video['id']
            title = video['snippet']['title']
            url = f"https://www.youtube.com/watch?v={vid}"
            writer.writerow([url, title])
    
    print(f"Data written to {OUTPUT_FILE}")
