import pandas as pd
import json
from datetime import timedelta

# Paths
TRANSCRIPT_PATH = "../clustering/intermediate_data/clustered_embeddings.csv"
TOPIC_CSV_PATH = "../topic_identification/top2vec_clustered_topics.csv"
TITLES_JSON_PATH = "../llm-check/titles_qwen(top2vec).json"
OUTPUT_JSON_PATH = "final_timestamps_titles_filled.json"

# Utilities
def time_to_seconds(t):
    parts = list(map(int, t.strip().split(":")))
    return parts[0]*60 + parts[1] if len(parts) == 2 else parts[0]*3600 + parts[1]*60 + parts[2]

def seconds_to_hms(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

# Load data
df_transcript = pd.read_csv(TRANSCRIPT_PATH)
df_transcript = df_transcript[df_transcript["text"].notnull()].dropna(subset=["start", "end"])
df_topics = pd.read_csv(TOPIC_CSV_PATH)

with open(TITLES_JSON_PATH, "r", encoding="utf-8") as f:
    title_data = json.load(f)

# Identify used texts
used_texts = set()
for rep_docs in df_topics["representative_docs"]:
    for d in rep_docs.split(" ||| "):
        used_texts.add(d.strip())

# Prepare topic ranges
topic_ranges = []
for _, row in df_topics.iterrows():
    topic_id = str(row["topic_num"])
    rep_docs = row["representative_docs"].split(" ||| ")
    matched = df_transcript[df_transcript["text"].isin(rep_docs)]
    if matched.empty:
        continue
    starts = matched["start"].map(time_to_seconds)
    ends = matched["end"].map(time_to_seconds)
    start_sec = int(starts.min())
    end_sec = int(ends.max())
    title = title_data.get(f"topic_{topic_id}", {}).get("title", "Unknown Title")
    topic_ranges.append((start_sec, end_sec, title))

# Sort and merge
topic_ranges.sort(key=lambda x: x[0])
final_output = []
last_end = 0

for start, end, title in topic_ranges:
    # Fill gap if needed
    if start > last_end:
        gap_entries = df_transcript[
            df_transcript["start"].map(time_to_seconds).between(last_end, start - 1)
            & ~df_transcript["text"].isin(used_texts)
        ]
        if not gap_entries.empty:
            gap_start = gap_entries["start"].map(time_to_seconds).min()
            gap_end = gap_entries["end"].map(time_to_seconds).max()
            combined_text = " ".join(gap_entries["text"].tolist())[:100]  # limit preview
            final_output.append({
                "start_time": seconds_to_hms(gap_start),
                "end_time": seconds_to_hms(gap_end),
                "title": f"Unclustered Segment: {combined_text.strip()}"
            })
            last_end = gap_end + 1

    # Add clustered topic
    final_output.append({
        "start_time": seconds_to_hms(start),
        "end_time": seconds_to_hms(end),
        "title": title
    })
    last_end = end + 1

# Save result
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Final filled timeline saved to {OUTPUT_JSON_PATH}")
