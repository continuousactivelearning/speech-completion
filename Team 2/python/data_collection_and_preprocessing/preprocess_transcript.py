import os
import json
import re
import pandas as pd

INPUT_DIR = "./formatted_transcripts"
OUTPUT_CSV = "./transcript_csvs/preprocessed_transcripts.csv"
MIN_WORDS = 50


def clean_text(text):
    text = re.sub(r"[!?]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def combine_transcript_chunks(transcript, min_words=MIN_WORDS):
    combined = []
    temp_text = ""
    start_time = transcript[0]["start"] if transcript else "0:00"
    for segment in transcript:
        cleaned = clean_text(segment["text"])
        temp_text += " " + cleaned
        word_count = len(temp_text.strip().split())
        if word_count >= min_words:
            combined.append(
                {"start": start_time, "end": segment["end"], "text": temp_text.strip()}
            )
            temp_text = ""
            start_time = segment["end"]
    if temp_text.strip():
        combined.append(
            {
                "start": start_time,
                "end": transcript[-1]["end"],
                "text": temp_text.strip(),
            }
        )
    return combined


all_data = []

for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.endswith(".json"):
        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            transcript = data.get("transcript", [])
            course = data.get("course", "unknown")
            source = data.get("source", "unknown")
            merged_segments = combine_transcript_chunks(transcript)
            for chunk in merged_segments:
                all_data.append(
                    {
                        "file": filename,
                        "course": course,
                        "source": source,
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": chunk["text"],
                    }
                )

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved cleaned data to {OUTPUT_CSV}")
