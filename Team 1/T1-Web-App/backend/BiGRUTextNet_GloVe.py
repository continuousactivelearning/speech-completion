'''import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from completion_model import BiGRUTextNet

nltk.download("punkt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GloVe vectors
def load_glove(path="glove.6B.300d.txt"):
    glove = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    return glove

glove_vectors = load_glove()

# Encode input sentence using GloVe
def encode_glove(text):
    tokens = word_tokenize(text.lower())
    vecs = [glove_vectors[token] for token in tokens if token in glove_vectors]
    return torch.tensor(np.mean(vecs, axis=0) if vecs else np.zeros(300), dtype=torch.float32)

def main():
    model_name = "BiGRUTextNet_GloVe"
    model = BiGRUTextNet(300).to(device)

    try:
        model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device))
    except FileNotFoundError:
        print(f"Trained model weights not found for: {model_name}")
        return

    model.eval()
    print(f"\nLoaded model: {model_name}")

    while True:
        text = input("\nEnter a sentence (or type 'exit' to quit): ").strip()
        if text.lower() == 'exit':
            print("Exiting.")
            break

        # Handle empty or whitespace-only input to avoid random guess
        if not text:
            print("Predicted Completion: 0.00% (Empty input)")
            continue

        input_tensor = encode_glove(text).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        print(f"Predicted Completion: {output.item():.2f}%")

if __name__ == "__main__":
    main()'''
