import pandas as pd
import requests
import numpy as np

# =============== CONFIG ==================
INPUT_CSV = "ml-32m/tags.csv"
FIRST_5_FILE = "first_5.csv"
ALL_FILE = "all_embeddings.csv"

GEMINI_API_KEY = "YOUR_REAL_ACCESS_TOKEN"
GEMINI_URL = "https://ai.google.dev/gemini-api/embeddings"


# ========================================


def get_gemini_embedding(text):
    """Calls Gemini API for text embeddings."""
    if not isinstance(text, str) or text.strip() == "":
        return [0.0] * 512  # Return zero-vector if text is invalid

    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"text": text, "model": "gemini-text-embedding"}

    try:
        resp = requests.post(GEMINI_URL, json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            emb = data.get("embedding", [])
            if not emb:
                return [0.0] * 512  # Return zero-vector if embedding is empty
            return emb[:512] + [0.0] * (512 - len(emb))  # Pad/truncate to 512
        else:
            print(f"[ERROR] {resp.status_code}: {resp.text}")
            return [0.0] * 512
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        retur
