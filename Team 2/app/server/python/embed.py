import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Generate embeddings and clusters, save as JSON")
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output JSON")
args = parser.parse_args()

df = pd.read_csv(args.input)
df = df[df["text"].notnull() & df["text"].str.strip().astype(bool)]
print(f"Loaded {len(df)} rows")

print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

df["embedding"] = [e.tolist() for e in embeddings]

df.to_csv(args.output, index=False)
print(f"OK: saved {len(df)} rows to {args.output}")
