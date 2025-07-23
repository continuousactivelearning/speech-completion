# preprocess_completion_data.py

import os
import pandas as pd
import numpy as np
import torch
import nltk
nltk.download('punkt_tab')
from tqdm import tqdm
from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel,
    RobertaTokenizer, RobertaModel
)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure preprocessed folder exists
os.makedirs("preprocessed", exist_ok=True)

# Paths to CSVs
splits = {
    "train": "train_df.csv",
    "val": "val_df.csv",
    "test": "test_df.csv"
}

# ========== Process 1: BERT ==========
def process1():
    print("\n=== Running process1: BERT ===")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    def preprocess_text(text):
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    for split_name, path in splits.items():
        df = pd.read_csv(path)
        embeddings, labels = [], []
        for text, label in tqdm(zip(df["Sentence"], df["completion_percentage"]),
                                total=len(df), desc=f"Process1 {split_name}"):
            embeddings.append(preprocess_text(text))
            labels.append(label)
        np.savez(f"preprocessed/preprocessed1_{split_name}.npz", text=np.array(embeddings), labels=np.array(labels))


# ========== Process 2: DistilBERT ==========
def process2():
    print("\n=== Running process2: DistilBERT ===")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device).eval()

    def preprocess_text(text):
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    for split_name, path in splits.items():
        df = pd.read_csv(path)
        embeddings, labels = [], []
        for text, label in tqdm(zip(df["Sentence"], df["completion_percentage"]),
                                total=len(df), desc=f"Process2 {split_name}"):
            embeddings.append(preprocess_text(text))
            labels.append(label)
        np.savez(f"preprocessed/preprocessed2_{split_name}.npz", text=np.array(embeddings), labels=np.array(labels))


# ========== Process 3: GloVe ==========
def process3():
    print("\n=== Running process3: GloVe (avg) ===")
    glove_path = "glove.6B.300d.txt"

    def load_glove(path):
        glove = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec
        return glove

    glove = load_glove(glove_path)
    print(f"Loaded GloVe with {len(glove)} tokens.")

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        vecs = [glove[token] for token in tokens if token in glove]
        return np.mean(vecs, axis=0) if vecs else np.zeros(300)

    for split_name, path in splits.items():
        df = pd.read_csv(path)
        embeddings, labels = [], []
        for text, label in tqdm(zip(df["Sentence"], df["completion_percentage"]),
                                total=len(df), desc=f"Process3 {split_name}"):
            embeddings.append(preprocess_text(text))
            labels.append(label)
        np.savez(f"preprocessed/preprocessed3_{split_name}.npz", text=np.array(embeddings), labels=np.array(labels))


# ========== Process 4: Sentence-BERT ==========
def process4():
    print("\n=== Running process4: Sentence-BERT ===")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))

    def preprocess_text(text):
        return model.encode(text, convert_to_numpy=True)

    for split_name, path in splits.items():
        df = pd.read_csv(path)
        embeddings, labels = [], []
        for text, label in tqdm(zip(df["Sentence"], df["completion_percentage"]),
                                total=len(df), desc=f"Process4 {split_name}"):
            embeddings.append(preprocess_text(text))
            labels.append(label)
        np.savez(f"preprocessed/preprocessed4_{split_name}.npz", text=np.array(embeddings), labels=np.array(labels))


# ========== Process 5: RoBERTa ==========
def process5():
    print("\n=== Running process5: RoBERTa ===")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device).eval()

    def preprocess_text(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.squeeze().cpu().numpy()

    for split_name, path in splits.items():
        df = pd.read_csv(path)
        embeddings, labels = [], []
        for text, label in tqdm(zip(df["Sentence"], df["completion_percentage"]),
                                total=len(df), desc=f"Process5 {split_name}"):
            embeddings.append(preprocess_text(text))
            labels.append(label)
        np.savez(f"preprocessed/preprocessed5_{split_name}.npz", text=np.array(embeddings), labels=np.array(labels))


# ========== Main ==========
if __name__ == '__main__':
    process1()
    process2()
    process3()
    process4()
    process5()
