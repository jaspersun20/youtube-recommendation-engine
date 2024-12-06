import pandas as pd

file_path = "tags_with_embeddings.csv"

# Read the CSV, ignoring malformed lines
df = pd.read_csv(file_path, on_bad_lines="skip")

print("First 20 Rows:")
print(df.head(20))

# print("\nLast 20 Rows:")
# print(df.tail(20))
