import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from model_completion import BiGRUTextNet

nltk.download("punkt_tab")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load GloVe vectors ---
def load_glove(path="glove.6B.300d.txt"):
    glove = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    return glove

# --- Encode input sentence using GloVe ---
def encode_glove(text, glove_vectors):
    tokens = word_tokenize(text.lower())
    vecs = [glove_vectors[token] for token in tokens if token in glove_vectors]
    return torch.tensor(np.mean(vecs, axis=0) if vecs else np.zeros(300), dtype=torch.float32)

# --- Load BiGRU model ---
def load_bigru_model(model_path="models/BiGRUTextNet_GloVe.pt"):
    model = BiGRUTextNet(300).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- Predict Completion Percentage using BiGRU ---
def predict_bigru_completion(text, model, glove_vectors):
    if not text.strip():
        return 0.0
    input_tensor = encode_glove(text, glove_vectors).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()  # returns percentage value
