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

# Close the connection
cursor.close()
connection.close()
print("All data ingested and connection closed.")
