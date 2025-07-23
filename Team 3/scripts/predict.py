import sys
import json
import numpy as np
import joblib
import os
import time
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ruptures as rpt
import pandas as pd
import warnings
from bertopic import BERTopic
from build_ensemble import (
    SpeechCompletionPredictor,
    TextFeatureExtractor,
    EmbeddingTransformer,
    ClusteringTransformer
)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# === Transformers Compatibility Fix ===
def fix_transformers_compatibility():
    import transformers
    # Monkey patch to fix the attribute error
    if not hasattr(transformers.models.mpnet.configuration_mpnet.MPNetConfig, '_output_attentions'):
        transformers.models.mpnet.configuration_mpnet.MPNetConfig._output_attentions = False
    if not hasattr(transformers.models.bert.configuration_bert.BertConfig, '_output_attentions'):
        transformers.models.bert.configuration_bert.BertConfig._output_attentions = False

# fix before loading models
fix_transformers_compatibility()

# === Paths ===
ROOT = Path(__file__).resolve().parent.parent
model_rf_path = ROOT / "model" / "tuned_random_forest_model.pkl"
model_cluster_path = ROOT / "model" / "speech_completion_clustering_model.pkl"
model_topic_path = ROOT / "model" / "final_bertopic_model"
weights_path = ROOT / "model" / "ensemble_weights.json"

# === Logging ===
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)
    

# === Load Models ===
try:
    rf_model = joblib.load(model_rf_path)
    log("Loaded Random Forest model.")
except Exception as e:
    log(f"Failed to load RF model: {e}")
    sys.exit(1)

try:
    cluster_model = joblib.load(model_cluster_path)
    log("Loaded clustering model.")
except Exception as e:
    log(f"Failed to load clustering model: {e}")
    sys.exit(1)

try:
    topic_model = BERTopic.load(model_topic_path)
    log("Loaded topic modeling (BERTopic) model.")
except Exception as e:
    log(f"Failed to load BERTopic model: {e}")
    sys.exit(1)

try:
    with open(weights_path) as f:
        weights = json.load(f)
        w_rf = weights.get("w_rf", 1.0)
        w_cluster = weights.get("w_cluster", 0.0)
        w_topic = weights.get("w_topic", 0.0)
    log(f"Loaded ensemble weights: w_rf={w_rf}, w_cluster={w_cluster}, w_topic={w_topic}")
except:
    log("Failed to load ensemble weights. Defaulting to w_rf=1.0, w_cluster=0.0, w_topic=0.0")
    w_rf, w_cluster, w_topic = 1.0, 0.0, 0.0

# === Sentence Embedder ===
try:
    embedder = SentenceTransformer("all-mpnet-base-v2")
    log("Loaded sentence transformer: all-mpnet-base-v2")
except Exception as e:
    log(f"Failed to load embedder: {e}")
    sys.exit(1)

# === Preprocessing ===
def preprocess_text(text):
    import re
    chunks = re.split(r'[.!?]', text)
    return [s.strip() for s in chunks if len(s.strip()) > 3]

# === Enhanced Features ===
def extract_enhanced_features(embeddings):
    if len(embeddings) < 2:
        return [0] * 10

    novelties = [1 - cosine_similarity([embeddings[i]], [embeddings[i - 1]])[0][0] for i in range(1, len(embeddings))]
    if not novelties:
        return [0] * 10

    mean_nov = np.mean(novelties)
    var_nov = np.var(novelties)
    max_nov = np.max(novelties)
    min_nov = np.min(novelties)
    trend = np.polyfit(range(len(novelties)), novelties, 1)[0]
    recent = np.mean(novelties[-min(5, len(novelties)):])

    # Change point detection
    cps = []
    try:
        novelty_array = np.array(novelties).reshape(-1, 1)
        model_cp = rpt.Pelt(model="l2").fit(novelty_array)
        cps = model_cp.predict(pen=3)[:-1]
    except:
        pass

    num_cps = len(cps)
    cp_density = num_cps / len(embeddings)
    recent_cp_activity = sum(1 for cp in cps if cp > 0.7 * len(embeddings)) / max(1, len(embeddings) - int(0.7 * len(embeddings)))
    cp_recency = (len(embeddings) - max(cps)) / len(embeddings) if cps else 0

    return [mean_nov, var_nov, max_nov, min_nov, trend, recent, num_cps, cp_density, recent_cp_activity, cp_recency]

# === Topic Distribution ===
def extract_topic_features(chunks):
    try:
        _, probs = topic_model.transform(chunks)
        topic_means = np.mean(probs, axis=0)
        return topic_means.tolist()
    except Exception as e:
        log(f"Topic feature extraction failed: {e}")
        return [0.0] * topic_model.get_topic_freq().shape[0]

# === Prediction ===
def predict_progress(text):
    chunks = preprocess_text(text)

    if len(chunks) < 5:
        est = min(100, (len(chunks) / 40) * 100)
        return round(est, 2)

    try:
        embeddings = embedder.encode(chunks, show_progress_bar=False)
    except Exception as e:
        log(f"Embedding failed: {e}")
        est = min(100, (len(chunks) / 40) * 100)
        return round(est, 2)

    # Random Forest
    rf_features = extract_enhanced_features(embeddings)
    rf_pred = rf_model.predict([rf_features])[0]

    # Clustering
    cluster_pred = 0
    if w_cluster > 0:
        try:
            X_df = pd.DataFrame({"Chunk": chunks, "Speech_ID": ["demo"] * len(chunks)})
            cluster_pred = cluster_model.predict(X_df)[0]
        except Exception as e:
            log(f"Clustering prediction failed: {e}")
            cluster_pred = 0

    # Topic Model
    topic_pred = 0
    if w_topic > 0:
        try:
            topic_features = extract_topic_features(chunks)
            topic_pred = np.dot(topic_features, np.linspace(30, 90, len(topic_features)))
        except Exception as e:
            log(f"Topic prediction failed: {e}")
            topic_pred = 0

    final_pred = w_rf * rf_pred + w_cluster * cluster_pred + w_topic * topic_pred
    return round(min(max(final_pred, 0), 100), 2)

# === CLI ===
if __name__ == "__main__":
    try:
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        text = data.get("transcript", "").strip()

        if not text:
            result = {"error": "Empty transcript provided"}
        else:
            prediction = predict_progress(text)
            result = {
                "prediction": prediction,
                "metadata": {
                    "length": len(text),
                    "timestamp": datetime.now().isoformat()
                }
            }

        print(json.dumps(result))

    except Exception as e:
        log(f"Error: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)