import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine

def load_movies_from_db():
    """
    Connects to your RDS MySQL instance, reads the 'movies' table,
    returns a DataFrame that includes columns like:
        - movieId (primary key)
        - imdb_rating (JSON string with "imdbRating" etc.)
        - genres (you likely also store 'genres' in the DB)
        - ... possibly other columns (title, year, etc.)
    """
    db_host = "movie-rec-db.czq8aoqqc0lw.us-east-2.rds.amazonaws.com"
    db_port = 3306
    db_user = "admin"
    db_password = "Jaspersun2001"
    db_name = "movie-rec-db"

    # Create SQLAlchemy engine
    engine = create_engine(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    # Read entire 'movies' table
    query = "SELECT * FROM movies;"
    movies_df = pd.read_sql(query, con=engine)
    print("[INFO] Loaded 'movies' table from DB. Shape =", movies_df.shape)
    return movies_df


def parse_imdb_rating(imdb_rating_json_str):
    """
    Given a JSON string like:
       {"imdbRating": "8.3", "Title": "Toy Story", ...}
    we extract just the "imdbRating".
    If not found or invalid, return None.
    """
    if not isinstance(imdb_rating_json_str, str):
        return None
    try:
        data = json.loads(imdb_rating_json_str)
        return data.get("imdbRating", None)  # e.g. "8.3"
    except:
        return None


import torch
from transformers import AutoTokenizer, AutoModel

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


def compute_movie_genre_embeddings(movies_df):
    """
    For each movie in movies_df, compute a 32D embedding by averaging
    the embeddings of its genres. We'll skip "imax".
    Returns a dict: movieId -> comma-separated 32D string
    """
    # 3.1) Build genre->embedding dictionary
    known_genres_raw = [
        "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western", "(no genres listed)"
        # no "IMAX"
    ]
    known_genres = [g.lower() for g in known_genres_raw]

    genre_to_emb = {}
    for g in known_genres:
        genre_to_emb[g] = get_text_embedding_32(g)

    # 3.2) Build a dictionary movieId -> 32D array
    movie_to_emb = {}
    for idx, row in movies_df.iterrows():
        mid = row["movieId"]
        raw_genres = row.get("genres", None)
        if not pd.notna(raw_genres):
            raw_genres = "(no genres listed)"

        # parse & skip imax
        genres_list = [g.strip().lower() for g in raw_genres.split("|") if g.strip()]
        genres_list = [g for g in genres_list if g != "imax"]

        if not genres_list:
            genres_list = ["(no genres listed)"]

        # gather recognized embeddings
        emb_list = []
        for g in genres_list:
            # unify "film noir" -> "film-noir"
            if g == "film noir":
                g = "film-noir"

            if g in genre_to_emb:
                emb_list.append(genre_to_emb[g])
            else:
                emb_list.append(genre_to_emb["(no genres listed)"])

        # average across recognized
        if len(emb_list) > 0:
            emb = np.mean(emb_list, axis=0)
        else:
            emb = np.zeros(32, dtype=np.float32)

        # store as a comma-separated string
        emb_str = ",".join(map(str, emb))
        movie_to_emb[mid] = emb_str

    return movie_to_emb



def main():
    print("[INFO] Starting add-on script...")

    # ---------------------------------------------------------
    # A) Load 'movies' from your DB & parse IMDB
    # ---------------------------------------------------------
    movies_df = load_movies_from_db()  # has columns: movieId, imdb_rating, genres, ...
    # parse imdb_rating JSON -> "imdb"
    movies_df["imdb"] = movies_df["imdb_rating"].apply(parse_imdb_rating)
    print("[INFO] Sample imdb values:", movies_df["imdb"].head(5).tolist())

    # ---------------------------------------------------------
    # B) Compute (or load) 32D genre embeddings for each movie
    # ---------------------------------------------------------
    movie_to_emb = compute_movie_genre_embeddings(movies_df)
    # store as 'movie_genre_embedding'
    movies_df["movie_genre_embedding"] = movies_df["movieId"].apply(lambda m: movie_to_emb[m])

    # ---------------------------------------------------------
    # C) Load your user-level features
    #    [ userId, avg_user_tag, user_like_genre ]
    # ---------------------------------------------------------
    user_features_df = pd.read_csv("user_with_genre_like_collapsed.csv")
    # ensure userId is int
    user_features_df["userId"] = user_features_df["userId"].astype(int)

    print("[INFO] user_features_df shape =", user_features_df.shape)

    # ---------------------------------------------------------
    # D) Load ratings.csv -> label yes/no
    # ---------------------------------------------------------
    ratings_df = pd.read_csv("../ml-32m/ratings.csv", encoding="latin-1")
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["label"] = ratings_df["rating"].apply(lambda r: "yes" if r>=4 else "no")
    print("[INFO] ratings_df shape =", ratings_df.shape)

    # ---------------------------------------------------------
    # E) Merge user_features + ratings on userId
    #    (this will replicate user features for each movie the user rated)
    # ---------------------------------------------------------
    user_ratings_df = pd.merge(ratings_df, user_features_df, on="userId", how="inner")
    print("[INFO] user_ratings_df shape =", user_ratings_df.shape)

    # ---------------------------------------------------------
    # F) Merge with movies_df on movieId -> final
    # ---------------------------------------------------------
    final_merged_df = pd.merge(user_ratings_df, movies_df, on="movieId", how="left")
    print("[INFO] final_merged_df shape =", final_merged_df.shape)

    # rename label -> "yes/no"
    final_merged_df.rename(columns={"label": "yes/no"}, inplace=True)

    # columns we want in final output
    output_cols = [
        "userId",
        "avg_user_tag",
        "user_like_genre",
        "imdb",                  # from JSON
        "movie_genre_embedding", # 32D vector
        "yes/no"
    ]
    final_output_df = final_merged_df[output_cols].copy()

    # ---------------------------------------------------------
    # G) Save final CSV
    # ---------------------------------------------------------
    out_file = "user_movie_training_dataset.csv"
    final_output_df.to_csv(out_file, index=False)
    print("[INFO] Done! The final dataset has columns:")
    print(", ".join(output_cols))
    print("[INFO] Rows:", len(final_output_df))
    print(f"[INFO] Saved to '{out_file}'.")

if __name__ == "__main__":
    main()
