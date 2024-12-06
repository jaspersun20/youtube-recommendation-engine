from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch

# Load pretrained DistilBERT model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Load tags data
tags_df = pd.read_csv('ml-32m/tags.csv')

# Function to generate embeddings for a tag
def get_tag_embedding(tag):
    # Ensure the tag is a string
    if not isinstance(tag, str):
        tag = ""
    # Tokenize and process tag
    inputs = tokenizer(tag, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling of the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Process tags in real time and append to the file
output_file = "tags_with_embeddings.csv"
with open(output_file, "w") as f:
    # Write the header
    f.write("userId,movieId,tag,timestamp,embedding\n")
    for _, row in tags_df.iterrows():
        try:
            tag_embedding = get_tag_embedding(row["tag"])
            embedding_str = ",".join(map(str, tag_embedding))  # Convert embedding to CSV format
            # Write row with embedding
            f.write(f"{row['userId']},{row['movieId']},{row['tag']},{row['timestamp']},{embedding_str}\n")
        except Exception as e:
            print(f"Error processing tag: {row['tag']}, skipping. Error: {e}")
