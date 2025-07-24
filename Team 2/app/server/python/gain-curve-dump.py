import pandas as pd
import json
import os
import random
import numpy as np
from scipy.optimize import curve_fit
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import argparse

nltk.download('stopwords')
nltk.download('punkt')

parser = argparse.ArgumentParser(description="Dump gain curves for sampled transcripts")
parser.add_argument("--input", required=True, help="Path to input CSV file")
parser.add_argument("--output", required=True, help="Path to output JSON file")
parser.add_argument("--samples", type=int, default=10, help="Number of transcripts to sample")
parser.add_argument("--dataset_size", type=int, default=2400, help="Total dataset size")
parser.add_argument("--stopword_weight", type=float, default=0.3, help="Stopword weight")

args = parser.parse_args()

stop_words = set(stopwords.words('english'))

df = pd.read_csv(args.input, on_bad_lines='skip')
df = df[df["transcript"].notnull() & df["transcript"].str.strip().astype(bool)]

def gain_model(x, a, b):
    return a * (1 - np.exp(-b * x))

random_indices = random.sample(range(args.dataset_size), args.samples)

output_data = []

for idx in random_indices:
    row = df.iloc[idx]
    text = row["transcript"]
    words = word_tokenize(text)
    words_alpha = [w.lower() for w in words if w.isalpha()]

    unique_words = set()
    stopword_count = 0
    total_counts = []
    gain_list = []

    for i, word in enumerate(words_alpha, 1):
        if word in stop_words:
            stopword_count += 1
        else:
            unique_words.add(word)

        gain = len(unique_words) - args.stopword_weight * stopword_count
        total_counts.append(i)
        gain_list.append(gain)

    x_data = np.array(total_counts)
    y_data = np.array(gain_list)

    try:
        popt, _ = curve_fit(gain_model, x_data, y_data, p0=[max(y_data), 0.001], maxfev=5000)
        a, b = popt
        y_target = 0.95 * a
        t_saturation = -np.log(1 - y_target / a) / b

        x_fit = np.linspace(1, x_data[-1], 200)
        y_fit = gain_model(x_fit, *popt)

        result = {
            "index": idx,
            "x_data": x_data.tolist(),
            "y_data": y_data.tolist(),
            "fitted_curve": {
                "x": x_fit.tolist(),
                "y": y_fit.tolist()
            },
            "t_saturation": t_saturation,
            "meta": {
                "total_words": len(words_alpha),
                "unique_words": len(unique_words),
                "stopwords": stopword_count
            }
        }

    except Exception as e:
        # if curve fit fails
        result = {
            "index": idx,
            "x_data": x_data.tolist(),
            "y_data": y_data.tolist(),
            "fitted_curve": None,
            "t_saturation": None,
            "meta": {
                "total_words": len(words_alpha),
                "unique_words": len(unique_words),
                "stopwords": stopword_count
            },
            "error": str(e)
        }

    output_data.append(result)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} gain curves to {args.output}")