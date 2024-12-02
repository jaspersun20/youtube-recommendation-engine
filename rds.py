import mysql.connector
import pandas as pd

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
# Create and use the database with backticks
cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
cursor.execute(f"USE `{db_name}`")

# Define schemas for tables
schemas = [
    """
    CREATE TABLE IF NOT EXISTS movies (
        movieId INT PRIMARY KEY,
        title VARCHAR(255),
        genres VARCHAR(255)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ratings (
        userId INT,
        movieId INT,
        rating FLOAT,
        timestamp BIGINT,
        PRIMARY KEY (userId, movieId),
        FOREIGN KEY (movieId) REFERENCES movies(movieId)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        userId INT,
        movieId INT,
        tag VARCHAR(255),
        timestamp BIGINT,
        PRIMARY KEY (userId, movieId, tag),
        FOREIGN KEY (movieId) REFERENCES movies(movieId)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS links (
        movieId INT PRIMARY KEY,
        imdbId INT,
        tmdbId INT,
        FOREIGN KEY (movieId) REFERENCES movies(movieId)
    )
    """
]

for schema in schemas:
    cursor.execute(schema)

print("Tables created successfully.")


# Load movies.csv
movies_df = pd.read_csv('ml-32m/movies.csv')
# Insert data into the movies table in batches
try:
    print("Starting ingestion of movies.csv...")
    movies_data = movies_df.values.tolist()
    insert_query = "INSERT IGNORE INTO movies (movieId, title, genres) VALUES (%s, %s, %s)"
    cursor.executemany(insert_query, movies_data)
    connection.commit()
    print("Data successfully ingested into the movies table.")
except Exception as e:
    print(f"Error while inserting data: {e}")

# Ingest `links.csv` with NaN handling
try:
    print("Starting ingestion of links.csv...")
    links_df = pd.read_csv('ml-32m/links.csv')
    # Fill NaN with a placeholder value or None for SQL NULL
    links_df.fillna(value={'imdbId': 0, 'tmdbId': 0}, inplace=True)
    links_data = links_df.values.tolist()
    insert_links_query = "INSERT IGNORE INTO links (movieId, imdbId, tmdbId) VALUES (%s, %s, %s)"
    cursor.executemany(insert_links_query, links_data)
    connection.commit()
    print("Data successfully ingested into the links table.")
except Exception as e:
    print(f"Error while inserting data into links: {e}")

# Ingest `tags.csv` with smaller batches and NaN handling
try:
    print("Starting ingestion of tags.csv...")
    tags_df = pd.read_csv('ml-32m/tags.csv')
    # Replace NaN values with None for SQL NULL
    tags_df.fillna(value={'tag': 'unknown'}, inplace=True)
    tags_data = tags_df.values.tolist()
    insert_tags_query = "INSERT IGNORE INTO tags (userId, movieId, tag, timestamp) VALUES (%s, %s, %s, %s)"
    batch_size = 1000  # Reduce batch size
    for i in range(0, len(tags_data), batch_size):
        cursor.executemany(insert_tags_query, tags_data[i:i + batch_size])
        connection.commit()
    print("Data successfully ingested into the tags table.")
except Exception as e:
    print(f"Error while inserting data into tags: {e}")

# Ingest `ratings.csv` in chunks
try:
    chunk_size = 10000
    for chunk in pd.read_csv('ml-32m/ratings.csv', chunksize=chunk_size):
        ratings_data = chunk.values.tolist()
        insert_ratings_query = "INSERT IGNORE INTO ratings (userId, movieId, rating, timestamp) VALUES (%s, %s, %s, %s)"
        cursor.executemany(insert_ratings_query, ratings_data)
        connection.commit()
        print(f"Chunk of size {len(ratings_data)} successfully ingested into the ratings table.")
    print("All data successfully ingested into the ratings table.")
except Exception as e:
    print(f"Error while inserting data into ratings: {e}")

# Close the connection
cursor.close()
connection.close()
print("All data ingested and connection closed.")
