import pandas as pd
import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Constants ---
EMBED_DIM = 384
NUM_CLASSES = 20
BUCKET_WIDTH = 100 / NUM_CLASSES

# --- Argparse Setup ---
parser = argparse.ArgumentParser(
    description="Predict average progress using LSTM model"
)
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output file")
args = parser.parse_args()

# --- Load CSV ---
df = pd.read_csv(args.input)
df = df[df["embedding"].notnull() & df["embedding"].str.strip().astype(bool)]
df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x), dtype=np.float32))
df = df.sort_values("start_sec")

# --- Prepare Sequence ---
sequence = [row for row in df["embedding"]]
X = [sequence]
X_padded = pad_sequences(X, dtype="float32", padding="post")

# --- Load Model & Predict ---
model = keras.models.load_model("progress_lstm_model")
predicted_class = model.predict(X_padded).argmax(axis=1)[0]
predicted_percent = (predicted_class + 0.5) * BUCKET_WIDTH

# --- Save to file ---
with open(args.output, "w") as f:
    f.write(f"{predicted_percent}")
