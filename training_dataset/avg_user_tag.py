import pandas as pd

# 1) Set the path to your CSV file (adjust as needed)
file_path = "../all_embeddings_32d.csv"

# 2) Define column names
#    We have 4 known columns: userId, movieId, tag, timestamp
#    Then 32 embedding columns, named embedding_0 through embedding_31
columns = (
        ["userId", "movieId", "tag", "timestamp"]
        + [f"embedding_{i}" for i in range(32)]
)

# 3) Read the file in chunks, because the dataset might be large
chunk_size = 100_000  # Adjust this based on memory constraints
chunks = pd.read_csv(
    file_path,
    sep=",",  # <-- Change to "\t" if your file is tab-delimited
    names=columns,  # <-- Use the manually-defined column names
    header=None,  # <-- Important, because there's no real header in the CSV
    chunksize=chunk_size,
    on_bad_lines="skip"  # Ignore malformed lines
)

# 4) List out the embedding columns
embedding_columns = [col for col in columns if col.startswith("embedding_")]

# 5) Create a list to store partial results from each chunk
user_embeddings = []

# 6) Process each chunk
for chunk in chunks:
    # Convert embedding columns to numeric (they might be read as strings)
    chunk[embedding_columns] = chunk[embedding_columns].apply(pd.to_numeric, errors="coerce")

    # Group by userId to compute the mean (averaging each of the 32 dimensions)
    chunk_avg = chunk.groupby("userId")[embedding_columns].mean().reset_index()

    # Append to our list
    user_embeddings.append(chunk_avg)

# 7) Concatenate all partial means, and group again (in case userIds span multiple chunks)
final_embeddings = pd.concat(user_embeddings).groupby("userId")[embedding_columns].mean().reset_index()

# 8) Save the final averaged embeddings for each userId to a new CSV
output_file = "avg_user_embeddings_32d.csv"
final_embeddings.to_csv(output_file, index=False)

print(f"✅ Done! The averaged 32D embeddings per userId are in '{output_file}'.")
