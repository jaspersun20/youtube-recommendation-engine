import mysql.connector

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

# Drop the `links` table if it exists
print("Dropping the `links` table if it exists...")
try:
    cursor.execute("DROP TABLE IF EXISTS links;")
    connection.commit()
    print("`links` table dropped successfully.")
except Exception as e:
    print(f"Error dropping the `links` table: {e}")

# Close the connection
cursor.close()
connection.close()
print("Operation completed.")
