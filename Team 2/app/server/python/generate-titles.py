import pandas as pd
import requests
import numpy as np
import argparse
from tqdm import tqdm
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from hashlib import md5

# --- ARGS SETUP ---
parser = argparse.ArgumentParser(
    description="Incremental topic detection + title generation"
)
parser.add_argument("--input", required=True, help="Path to input CSV with embeddings")
parser.add_argument("--output", required=True, help="Path to output CSV with titles")
args = parser.parse_args()

# --- CONFIG ---
OLLAMA_MODEL = "gemma:2b"
TOP_N_KEYWORDS = 5
SIMILARITY_THRESHOLD = 0.5
MAX_INPUT_TOKENS = 1000

# --- SETUP ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedding_model)
title_cache = {}


def parse_embedding(embedding_str):
    return np.array(eval(embedding_str))


def extract_keywords(text, top_n=TOP_N_KEYWORDS):
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words="english")
    return [kw for kw, _ in keywords]


def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


def generate_title(keywords):
    prompt = f"""You are a title generator.

Given the following keywords, generate a concise lecture topic title.

- Title must be 2 to 5 words
- No punctuation
- No complete sentences
- No repetition of keywords
- No quotes, no explanations
- Do not write anything else
- Just the title

Keywords: {', '.join(keywords)}
Title:"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 20},
            },
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # --- Clean LLM response ---
        for line in raw.splitlines():
            line = line.strip().strip('"').strip("'").strip("`")
            # Pick the first valid short phrase (2–5 words)
            if 2 <= len(line.split()) <= 5:
                return line

        raise ValueError("No valid short line found")

    except Exception as e:
        print(f"[Error generating title] {e}")
        return f"Error - {type(e).__name__}"


def get_cached_title(keywords):
    key = md5(" ".join(sorted(keywords)).encode()).hexdigest()
    if key in title_cache:
        return title_cache[key]
    title = generate_title(keywords)
    title_cache[key] = title
    return title


# --- LOAD DATA ---
df = pd.read_csv(args.input)
df = df[df["text"].notnull() & df["embedding"].notnull()]
df = df.sort_values("start").reset_index(drop=True)

# --- PROCESS ---
results = []
i = 0
n = len(df)

while i < n:
    seed_rows = df.iloc[i : i + 5]
    topic_start = seed_rows.iloc[0]["start"]

    topic_keywords = set()
    topic_vec = None

    for _, row in seed_rows.iterrows():
        text = row["text"].strip()
        emb = parse_embedding(row["embedding"])
        keywords = extract_keywords(text)
        topic_keywords.update(keywords)
        topic_vec = emb if topic_vec is None else (topic_vec + emb) / 2

    title = get_cached_title(list(topic_keywords))
    print(title)
    results.append({"start": topic_start, "title": title})

    i += 5

    while i < n:
        row = df.iloc[i]
        text = row["text"].strip()
        emb = parse_embedding(row["embedding"])
        similarity = compute_similarity(topic_vec, emb)

        if similarity >= SIMILARITY_THRESHOLD:
            keywords = extract_keywords(text)
            topic_keywords.update(keywords)
            topic_vec = (topic_vec + emb) / 2
            i += 1
        else:
            break


if results:
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\nDone! Titles saved to {args.output}")
else:
    print("⚠️ No titles generated — empty result.")
