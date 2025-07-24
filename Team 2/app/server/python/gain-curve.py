import pandas as pd
import spacy
import random
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Dump gain curves for sampled text rows")
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output JSON")
parser.add_argument("--samples", type=int, default=10, help="Number of random samples")
parser.add_argument("--stopword-weight", type=float, default=0.3, help="Stopword penalty weight")
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

df = pd.read_csv(args.input)
df = df[df["text"].notnull() & df["text"].str.strip().astype(bool)]

files = df["file"].unique()

if args.samples < len(files):
    files = random.sample(list(files), args.samples)

# random_indices = random.sample(range(len(df)), args.samples)

output = {}

# for idx in random_indices:
#     row = df.iloc[idx]
#     text = row["text"]

#     doc = nlp(text)
#     words_alpha = [token.text.lower() for token in doc if token.is_alpha]

#     unique_words = set()
#     stopword_count = 0
#     gain_list = []

#     for i, word in enumerate(words_alpha, 1):
#         if word in stop_words:
#             stopword_count += 1
#         else:
#             unique_words.add(word)

#         gain = len(unique_words) - args.stopword_weight * stopword_count
#         gain_list.append({"position": i, "gain": gain})

#     row_output = {
#         "file": row["file"],
#         "cluster": int(row["cluster"]),
#         "text": text,
#         "data": gain_list,
#         "summary": {
#             "total_words": len(words_alpha),
#             "unique_words": len(unique_words),
#             "stopwords": stopword_count
#         }
#     }

#     output.append(row_output)

for file_name in files:
    df_file = df[df["file"] == file_name]

    combined_text = " ".join(df_file["text"].tolist())

    doc = nlp(combined_text)
    words_alpha = [token.text.lower() for token in doc if token.is_alpha]

    unique_words = set()
    stopword_count = 0
    gain_list = []

    for i, word in enumerate(words_alpha, 1):
        if word in stop_words:
            stopword_count += 1
        else:
            unique_words.add(word)

        gain = len(unique_words) - args.stopword_weight * stopword_count
        gain_list.append({"position": i, "gain": gain})

    file_output = {
        "text" : combined_text,
        "data" : gain_list
    }

    output[file_name] = file_output
    

Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

# print(f"OK: saved {args.output} with {args.samples} samples")
print(f"OK: saved {args.output} with {len(files)} files")