import pandas as pd


def csv_to_parquet(
        csv_input="user_movie_training_dataset.csv",
        parquet_output="user_movie_training_dataset.parquet"
):
    """
    Reads the CSV (which has columns:
      userId,
      avg_user_tag (32D string),
      user_like_genre (32D string, if you want to use it),
      imdb (float),
      movie_genre_embedding (32D string),
      yes/no (label)
    and saves it as a Parquet file.
    """
    df = pd.read_csv(csv_input)
    # Just to confirm: the CSV has "yes/no" column. We'll keep it.
    # Then we'll parse it in our Dataset later.

    df.to_parquet(parquet_output, index=False)
    print(f"[INFO] Wrote Parquet: {parquet_output} (rows={len(df)})")
csv_input = "user_movie_training_dataset.csv"
parquet_output = "user_movie_training_dataset.parquet"

# 1) Convert CSV => Parquet
csv_to_parquet(csv_input, parquet_output)