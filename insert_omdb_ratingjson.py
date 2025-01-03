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

# Drop all foreign key constraints dynamically
print("Dropping foreign key constraints...")
try:
    cursor.execute("""
    SELECT CONSTRAINT_NAME, TABLE_NAME
    FROM information_schema.KEY_COLUMN_USAGE
    WHERE REFERENCED_TABLE_NAME = 'movies' AND TABLE_SCHEMA = DATABASE();
    """)
    constraints = cursor.fetchall()

    if constraints:
        for constraint_name, table_name in constraints:
            drop_query = f"ALTER TABLE {table_name} DROP FOREIGN KEY {constraint_name};"
            cursor.execute(drop_query)
            print(f"Dropped foreign key {constraint_name} from {table_name}.")
        connection.commit()
    else:
        print("No foreign key constraints referencing the `movies` table found.")
except Exception as e:
    print(f"Error dropping foreign key constraints: {e}")

# Drop and recreate the `movies` table
print("Dropping and recreating the `movies` table...")
try:
    cursor.execute("DROP TABLE IF EXISTS movies;")
    connection.commit()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        movieId INT PRIMARY KEY,
        title VARCHAR(255),
        genres VARCHAR(255),
        imdb_rating JSON,
        imdbId VARCHAR(20),
        tmdbId VARCHAR(20)
    );
    """)
    connection.commit()
    print("`movies` table recreated.")
except Exception as e:
    print(f"Error recreating the `movies` table: {e}")

# Load and insert test data into the `movies` table
print("Loading and inserting data into `movies`...")
try:
    movies_df = pd.read_csv('ml-32m/movies.csv', encoding='latin1')  # Replace this with your actual test CSV file path
    movies_df.fillna({'imdbId': '0', 'tmdbId': '0'}, inplace=True)
    insert_query = """
    INSERT INTO movies (movieId, title, genres, imdbId, tmdbId)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor.executemany(insert_query, movies_df[['movieId', 'title', 'genres', 'imdbId', 'tmdbId']].values.tolist())
    connection.commit()
    print("Data successfully inserted into `movies`.")
except Exception as e:
    print(f"Error inserting data into `movies`: {e}")

# Fetch IMDb rating and update the `movies` table
def fetch_imdb_rating(imdb_id):
    imdb_id = imdb_id.zfill(7)  # Ensure IMDb ID is 7 characters long
    omdb_url = f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={omdb_api_key}"
    response = requests.get(omdb_url)
    if response.status_code == 200:
        data = response.json()
        if "Response" in data and data["Response"] == "True":
            return data
        else:
            print(f"No valid response for IMDb ID: {imdb_id}")
            return None
    else:
        print(f"OMDb API Error: {response.status_code}")
        return None

# Update IMDb ratings in JSON format
print("Fetching and updating IMDb ratings as JSON...")
try:
    for _, row in movies_df.iterrows():
        imdb_id = str(row['imdbId']).strip()
        if imdb_id != '0':  # Skip invalid IMDb IDs
            imdb_data = fetch_imdb_rating(imdb_id)
            if imdb_data is not None:
                try:
                    cursor.execute("""
                    UPDATE movies 
                    SET imdb_rating = %s 
                    WHERE imdbId = %s
                    """, (json.dumps(imdb_data), imdb_id))
                    connection.commit()
                    print(f"Updated IMDb rating for IMDb ID {imdb_id}")
                except Exception as e:
                    print(f"Database update error for IMDb ID {imdb_id}: {e}")
except Exception as e:
    print(f"Error updating IMDb ratings: {e}")

# Re-add foreign key constraints
print("Re-adding foreign key constraints...")
try:
    cursor.execute("""
    ALTER TABLE ratings ADD CONSTRAINT ratings_ibfk_1 FOREIGN KEY (movieId) REFERENCES movies(movieId);
    """)
    cursor.execute("""
    ALTER TABLE tags ADD CONSTRAINT tags_ibfk_1 FOREIGN KEY (movieId) REFERENCES movies(movieId);
    """)
    cursor.execute("""
    ALTER TABLE links ADD CONSTRAINT links_ibfk_1 FOREIGN KEY (movieId) REFERENCES movies(movieId);
    """)
    connection.commit()
    print("Foreign key constraints re-added.")
except Exception as e:
    print(f"Error re-adding foreign key constraints: {e}")

# Close the connection
cursor.close()
connection.close()
print("Operation completed.")
