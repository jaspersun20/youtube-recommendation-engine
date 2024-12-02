import boto3
import csv
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Constants
CSV_FILE_PATH = 'ml-32m/ml-youtube.csv'
BUCKET_NAME = 'movies-yitian'
THUMBNAILS_FOLDER = 'movies/111'
AWS_REGION = 'us-east-2'
# Boto3 S3 client setup
s3_client = boto3.client('s3', region_name=AWS_REGION)

def read_csv(file_path):
    video_ids = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        for row in csvreader:
            video_ids.append(row[0])
    return video_ids

def fetch_and_upload_thumbnail(video_id):
    thumbnail_types = ['0', '1', '2', '3']
    for thumb_type in thumbnail_types:
        thumbnail_url = f'https://img.youtube.com/vi/{video_id}/{thumb_type}.jpg'
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            # Upload thumbnail to S3
            s3_key = f'{THUMBNAILS_FOLDER}/{video_id}_{thumb_type}.jpg'
            s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=response.content)
            print(f'Uploaded {s3_key} to S3')
        else:
            print(f'Failed to fetch thumbnail {thumbnail_url}')

def process_video_ids(video_ids):
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        executor.map(fetch_and_upload_thumbnail, video_ids)

def main():
    video_ids = read_csv(CSV_FILE_PATH)
    # Split video ids into 4 parts for parallel processing
    split_size = len(video_ids) // 4
    parts = [video_ids[i:i + split_size] for i in range(0, len(video_ids), split_size)]

    start_time = time.time()

    # Run parallel tasks for each part
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_video_ids, parts)

    end_time = time.time()
    print(f'Time taken to download thumbnails: {end_time - start_time} seconds')

if __name__ == '__main__':
    main()
