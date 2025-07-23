# split_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Step 0: Load dataset in chunks to avoid MemoryError
chunks = []
chunksize = 10_000  # You can lower this to 5000 or increase if you have more memory

for chunk in pd.read_csv("combined_completion_all_videos.csv", chunksize=chunksize):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

# Step 1: Stratified split (80% train+val, 20% test) based on 'video_name'
train_val_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["video_name"],
    random_state=42,
    shuffle=True
)

# Step 2: Stratified split of train_val into 70% train and 30% val based on 'video_name'
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.30,
    stratify=train_val_df["video_name"],
    random_state=42,
    shuffle=True
)

# Keep only required columns
train_df = train_df[["Sentence", "completion_percentage"]]
val_df = val_df[["Sentence", "completion_percentage"]]
test_df = test_df[["Sentence", "completion_percentage"]]

# Save to CSV
train_df.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)
test_df.to_csv("test_df.csv", index=False)

# Print sizes
print(f"Total samples: {len(df)}")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
