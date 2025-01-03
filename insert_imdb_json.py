import mysql.connector
import requests
import pandas as pd
import json

# OMDb API Key
omdb_api_key = "6145e2e1"

# Database connection parameters
db_host = 'movie-rec-db.czq8aoqqc0lw.us-east-2.rds.amazonaws.com'
db_user = 'admin'
db_password = 'Jaspersun2001'
db_name = 'movie-rec-db'

# Connect to MySQL database
connection = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)
cursor = connection.cursor()

# Add the imdb_rating column and api id columns if they don't exist
def ensure_column_exists(column_name, column_type):
    cursor.execute(f"""
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'movies' AND COLUMN_NAME = '{column_name}';
    """)
    if cursor.fetchone()[0] == 0:
        cursor.execute(f"ALTER TABLE movies ADD COLUMN {column_name} {column_type};")
        connection.commit()

ensure_column_exists("imdb_rating", "JSON")
ensure_column_exists("imdbId", "VARCHAR(20)")
ensure_column_exists("tmdbId", "VARCHAR(20)")

# Function to fetch IMDb rating from OMDb API
def fetch_imdb_rating(imdb_id):
    while len(imdb_id)<8:
        imdb_id = "0" + imdb_id
    # print(imdb_id)
    omdb_url = f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={omdb_api_key}"
    response = requests.get(omdb_url)
    if response.status_code == 200:
        data = response.json()
        if "imdbRating" in data:
            return data
        else:
            print(f"IMDb Rating not found for ID: {imdb_id}")
            return {}
    else:
        print(f"OMDb API Error: {response.status_code}")
        return {}

# Load movies data to get IMDb and TMDb IDs
try:
    movies_df = pd.read_csv('ml-32m/movies.csv', encoding='latin1')
    movies_df.fillna(value={'imdbId': '0', 'tmdbId': '0'}, inplace=True)
except Exception as e:
    print(f"Error loading movies.csv: {e}")
    exit()

# Iterate over the movies and fetch IMDb ratings
for _, row in movies_df.iterrows():
    imdb_id = str(row['imdbId']).strip()
    tmdb_id = str(row['tmdbId']).strip()
    try:
        imdb_data = fetch_imdb_rating(imdb_id)
        if imdb_data:
            # Insert JSON data into the imdb_rating column
            cursor.execute(
                """
                UPDATE movies 
                SET imdb_rating = %s, imdbId = %s, tmdbId = %s 
                WHERE imdbId = %s;
                """,
                (json.dumps(imdb_data), imdb_id, tmdb_id, imdb_id)
            )
            connection.commit()
            print(f"Inserted IMDb data for movie: {row.get('title', 'Unknown Title')}")
    except Exception as e:
        print(f"Error processing movie {row.get('title', 'Unknown Title')} with IMDb ID {imdb_id}: {e}")

# Close the connection
cursor.close()
connection.close()
print("IMDb data insertion completed.")
