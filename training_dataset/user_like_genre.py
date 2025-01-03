import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------
# 1) DistilBERT Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()


@torch.no_grad()
def get_text_embedding_32(text: str) -> np.ndarray:
    """
    Takes a string, returns a 32D embedding from DistilBERT:
      1) Tokenize
      2) DistilBERT forward pass
      3) Mean-pool across seq_len
      4) Take first 32 dims from 768D
    """
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # shape [1, seq_len, 768]
    emb_768 = last_hidden.mean(dim=1).squeeze(0)  # shape [768]
    emb_32 = emb_768[:32].cpu().numpy()           # shape [32]
    return emb_32


def main():
    # ---------------------------------------------------------
    # 2) Define known genres (lowercased). We'll skip "imax".
    # ---------------------------------------------------------
    known_genres_raw = [
        "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western", "(no genres listed)"
    ]
    known_genres = [g.lower() for g in known_genres_raw]

    # Build genre->embedding dictionary
    genre_to_emb = {}
    for g in known_genres:
        emb_32 = get_text_embedding_32(g)
        genre_to_emb[g] = emb_32

    # ---------------------------------------------------------
    # 3) Load movies.csv, ignoring "imax"
    # ---------------------------------------------------------
    movies_df = pd.read_csv("../ml-32m/movies.csv", encoding="latin-1")
    movie_to_emb = {}

    for idx, row in movies_df.iterrows():
        mid = row["movieId"]
        raw_genres = row["genres"] if pd.notna(row["genres"]) else "(no genres listed)"
        genres_list = [g.strip().lower() for g in raw_genres.split("|") if g.strip()]
        genres_list = [g for g in genres_list if g != "imax"]

        if not genres_list:
            genres_list = ["(no genres listed)"]

        emb_list = []
        for g in genres_list:
            if g == "film noir":
                g = "film-noir"
            if g in genre_to_emb:
                emb_list.append(genre_to_emb[g])
            else:
                emb_list.append(genre_to_emb["(no genres listed)"])

        movie_to_emb[mid] = np.mean(emb_list, axis=0) if emb_list else np.zeros(32, dtype=np.float32)

    # ---------------------------------------------------------
    # 4) Load ratings.csv.
    # ---------------------------------------------------------
    ratings_df = pd.read_csv("../ml-32m/ratings.csv", encoding="latin-1")
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    above4_ratings_df = ratings_df[ratings_df["rating"] >= 4].copy()

    user_to_movie_embs = {}
    for idx, row in above4_ratings_df.iterrows():
        uid = row["userId"]
        mid = row["movieId"]
        if mid not in movie_to_emb:
            continue
        user_to_movie_embs.setdefault(uid, []).append(movie_to_emb[mid])

    user_to_genre_like_emb = {
        uid: np.mean(emb_list, axis=0) if emb_list else np.zeros(32, dtype=np.float32)
        for uid, emb_list in user_to_movie_embs.items()
    }

    # Ensure keys in user_to_genre_like_emb are np.float64
    user_to_genre_like_emb = {np.float64(k): v for k, v in user_to_genre_like_emb.items()}

    print("[DEBUG] Number of users with rating>=4 movie embeddings:", len(user_to_genre_like_emb))

    # ---------------------------------------------------------
    # 5) Load avg_user_embeddings_32d.csv
    # ---------------------------------------------------------
    avg_user_emb_df = pd.read_csv("avg_user_embeddings_32d.csv")

    # Clean userId column
    avg_user_emb_df = avg_user_emb_df[pd.to_numeric(avg_user_emb_df['userId'], errors='coerce').notnull()]
    avg_user_emb_df['userId'] = avg_user_emb_df['userId'].astype(np.float64)

    # Combine embedding columns into a single string
    embedding_cols = [f"embedding_{i}" for i in range(32)]
    avg_user_emb_df["avg_user_tag"] = avg_user_emb_df[embedding_cols].apply(
        lambda row: ",".join(map(str, row.values)), axis=1
    )

    # ---------------------------------------------------------
    # 6) Build user_like_genre col
    # ---------------------------------------------------------
    like_genre_list = []
    for uid in avg_user_emb_df["userId"]:
        if uid in user_to_genre_like_emb:
            arr = user_to_genre_like_emb[uid]
        else:
            arr = np.zeros(32, dtype=np.float32)
        like_genre_list.append(",".join(map(str, arr)))

    avg_user_emb_df["user_like_genre"] = like_genre_list

    # Keep only relevant columns
    final_df = avg_user_emb_df[["userId", "avg_user_tag", "user_like_genre"]]

    print("[DEBUG] final_df.shape =", final_df.shape)
    print(final_df.head())

    # ---------------------------------------------------------
    # 7) Save final CSV
    # ---------------------------------------------------------
    out_file = "user_with_genre_like_collapsed.csv"
    final_df.to_csv(out_file, index=False)

    print(f"[INFO] Done! The file '{out_file}' has columns:")
    print("  userId, avg_user_tag, user_like_genre")


if __name__ == "__main__":
    main()
