import pandas as pd
import requests

# Load tags data
tags_df = pd.read_csv('ml-32m/tags.csv')

# Your Gemini API key
gemini_api_key = "1031273837835-0t872n1vo9s9mat1kgh1g1lu2ikvbra2.apps.googleusercontent.com"


# Function to generate embeddings for a tag using Gemini API
def get_tag_embedding(tag):
    if not isinstance(tag, str):
        tag = ""

    # Make API request to Gemini
    url = "https://ai.google.dev/gemini-api/embeddings"
    headers = {"Authorization": f"Bearer {gemini_api_key}"}
    data = {"text": tag, "model": "gemini-text-embedding"}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["embedding"]  # Return the embedding array
    else:
        print(f"Error fetching embedding for tag '{tag}': {response.status_code} {response.text}")
        return [0] * 32  # Return a zero vector if API call fails


# Process tags in real time and append to the file
output_file = "tags_with_embeddings_gemini.csv"
with open(output_file, "w") as f:
    # Write the header
    f.write("userId,movieId,tag,timestamp,embedding\n")

    for _, row in tags_df.iterrows():
        try:
            # Generate embedding for the tag
            tag_embedding = get_tag_embedding(row["tag"])

            # Convert embedding to CSV format
            embedding_str = ",".join(map(str, tag_embedding))

            # Write the row with embedding
            f.write(f"{row['userId']},{row['movieId']},{row['tag']},{row['timestamp']},{embedding_str}\n")
        except Exception as e:
            print(f"Error processing tag: {row['tag']}, skipping. Error: {e}")
