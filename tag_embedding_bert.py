import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import math

# =============== CONFIG ===================
INPUT_CSV = "ml-32m/tags.csv"
FIRST_5_FILE = "first_5row111.csv"
ALL_FILE = "all_embeddings_32d.csv"\

# How many rows to process per batch
# Increase if you have enough GPU memory
BATCH_SIZE = 32

# =============== DISTILBERT LOADING =======
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()  # no dropout
model.to("cuda" if torch.cuda.is_available() else "cpu")  # move model to GPU if available


@torch.no_grad()
def get_tag_embeddings_batch(tag_list):
    """
    Given a list of N tags (strings), returns a list of N embeddings (768D each).
    We'll tokenize them in one batch to speed up inference,
    then do mean pooling.

    :param tag_list: list of strings
    :return: list of numpy arrays, each shape [768]
    """
    if not tag_list:
        return []
    # 1) Tokenize all tags as a batch
    inputs = tokenizer(tag_list, return_tensors="pt", truncation=True,
                       padding=True, max_length=64)  # tune max_length if needed

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    # 2) Forward pass
    outputs = model(**inputs)  # last_hidden_state shape: [batch_size, seq_len, hidden_size=768]
    last_hidden = outputs.last_hidden_state

    # 3) Mean pool each sequence (dim=1)
    # shape: [batch_size, hidden_size]
    emb_768 = last_hidden.mean(dim=1)

    # 4) Move back to CPU numpy
    emb_768_cpu = emb_768.cpu().numpy()  # shape: [batch_size, 768]

    return emb_768_cpu


def main():
    # =========== 1) Load the CSV ============
    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        print("[ERROR] The input CSV file is empty. Please check the file.")
        return
    if "tag" not in df.columns:
        print("[ERROR] The input CSV does not contain a 'tag' column.")
        return

    print(f"[INFO] Loaded {len(df)} rows from {INPUT_CSV}")
    print("[DEBUG] DataFrame columns:", list(df.columns))
    print("[DEBUG] df.head(5):\n", df.head(5))

    # ========== 2) Prepare Data ============
    user_ids, movie_ids, tags, timestamps = [], [], [], []

    for idx, row in df.iterrows():
        user_ids.append(str(row.get("userId", "")))
        movie_ids.append(str(row.get("movieId", "")))
        tag_text = row.get("tag", "")
        if pd.isna(tag_text) or not isinstance(tag_text, str) or tag_text.strip() == "":
            tag_text = "[EMPTY_TAG]"
        tags.append(tag_text)
        timestamps.append(str(row.get("timestamp", "")))

    total_rows = len(tags)
    print(f"[INFO] Starting batch inference on {total_rows} tags with batch_size={BATCH_SIZE}...")

    # ========== 3) Write Data ============
    with open(FIRST_5_FILE, "w", encoding="utf-8") as f5, open(ALL_FILE, "w", encoding="utf-8") as fall:
        f5.write("userId,movieId,tag,timestamp,embedding\n")
        fall.write("userId,movieId,tag,timestamp,embedding\n")
        first_5_count = 0

        start_idx = 0
        row_idx = 0

        while start_idx < total_rows:
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch_tags = tags[start_idx:end_idx]

            print(f"[DEBUG] Processing batch {start_idx}-{end_idx}...")

            emb_batch_768 = get_tag_embeddings_batch(batch_tags)
            print(f"[DEBUG] Received {len(emb_batch_768)} embeddings.")

            for i in range(len(emb_batch_768)):
                emb_768 = emb_batch_768[i]
                emb_32 = emb_768[:32]
                emb_32_str = ",".join(map(str, emb_32))

                try:
                    uid = user_ids[row_idx]
                    mid = movie_ids[row_idx]
                    tag_text = tags[row_idx]
                    ts = timestamps[row_idx]
                except IndexError:
                    print(f"[ERROR] Index out of range at row_idx={row_idx}.")
                    break

                fall.write(f"{uid},{mid},{tag_text},{ts},{emb_32_str}\n")

                if first_5_count < 5:
                    f5.write(f"{uid},{mid},{tag_text},{ts},{emb_32_str}\n")
                    print(f"[DEBUG] Row {row_idx} -> tag='{tag_text}'")
                    print(f"         32D embedding={emb_32}\n")
                    first_5_count += 1

                row_idx += 1

            # Explicit flush to ensure data is written
            fall.flush()
            f5.flush()
            start_idx += BATCH_SIZE

    print("[INFO] Done! Check:")
    print(f"  - {FIRST_5_FILE} for the first 5 processed rows.")
    print(f"  - {ALL_FILE} for all 32D embeddings.")


if __name__ == "__main__":
    main()
