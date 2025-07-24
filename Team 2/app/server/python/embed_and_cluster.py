import pandas as pd
import numpy as np
import argparse
import hdbscan
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from umap import UMAP

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

print("Reducing dimensionality...")
pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
pca_embeddings = pca.fit_transform(embeddings)

umap_embeddings = UMAP(n_components=20, random_state=42).fit_transform(pca_embeddings)

print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)

start = time.time()
cluster_labels = clusterer.fit_predict(umap_embeddings)
end = time.time()
print(f"Clustering done in {end - start:.2f} sec")

df["embedding"] = [e.tolist() for e in embeddings]
df["cluster"] = cluster_labels

df.to_csv(args.output, index=False)
print(f"OK: saved {len(df)} rows to {args.output}")
