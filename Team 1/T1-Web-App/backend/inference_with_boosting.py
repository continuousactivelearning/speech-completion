
import torch
import numpy as np
import nltk
from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel,
    RobertaTokenizer, RobertaModel
)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from completion_model import (
    TextRegressionNet,
    DualTextFusionNet,
    BiGRUTextNet,
    DualTextAttentionNet,
    GatedTextFusionNet,
    AveragedFusionNet
)
import joblib

nltk.download("punkt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def encode_bert(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

def encode_distilbert(text):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device).eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

def encode_roberta(text):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device).eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().cpu()

def encode_sbert(text):
    model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    return torch.tensor(model.encode(text), dtype=torch.float32)

def encode_glove(text):
    tokens = word_tokenize(text.lower())
    vecs = [glove_vectors[token] for token in tokens if token in glove_vectors]
    return torch.tensor(np.mean(vecs, axis=0) if vecs else np.zeros(300), dtype=torch.float32)

model_map = {
    # 1: ("TextRegressionNet_BERT", TextRegressionNet(768), [encode_bert]),
    # 2: ("TextRegressionNet_DistilBERT", TextRegressionNet(768), [encode_distilbert]),
    1: ("BiGRUTextNet_GloVe", BiGRUTextNet(300), [encode_glove]),
    # 4: ("TextRegressionNet_SBERT", TextRegressionNet(384), [encode_sbert]),
    # 5: ("TextRegressionNet_RoBERTa", TextRegressionNet(768), [encode_roberta]),
    # 6: ("DualTextFusionNet_BERT_RoBERTa", DualTextFusionNet(768, 768), [encode_bert, encode_roberta]),
    # 7: ("DualTextAttentionNet_SBERT_BERT", DualTextAttentionNet(384, 768), [encode_sbert, encode_bert]),
    # 8: ("GatedTextFusionNet_GloVe_RoBERTa", GatedTextFusionNet(300, 768), [encode_glove, encode_roberta]),
    2: ("AveragedFusionNet_GloVe_BERT", AveragedFusionNet(300, 768), [encode_glove, encode_bert]),
}

boosting_model_paths = {
    # 10: ("XGBoost_BERT", "boosting_models/XGBoost_BERT_final.pt", encode_bert),
    # 11: ("XGBoost_DistilBERT", "boosting_models/XGBoost_DistilBERT_final.pt", encode_distilbert),
    3: ("XGBoost_GloVe", "boosting_models/XGBoost_GloVe_final.pt", encode_glove),
    # 13: ("XGBoost_SBERT", "boosting_models/XGBoost_SBERT_final.pt", encode_sbert),
    # 14: ("XGBoost_RoBERTa", "boosting_models/XGBoost_RoBERTa_final.pt", encode_roberta),
}

def main():
    print("\nChoose a model for prediction:\n")
    for k, (name, _, _) in model_map.items():
        print(f"{k}: {name}")
    for k, (name, _, _) in boosting_model_paths.items():
        print(f"{k}: {name}")

    choice = int(input("\nEnter the model number (e.g., 1): ").strip())

    if choice in model_map:
        model_name, model_class, preprocessors = model_map[choice]
        model = model_class.to(device)

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

            inputs = [fn(text).unsqueeze(0).to(device) for fn in preprocessors]

            with torch.no_grad():
                output = model(*inputs)

            print(f"Predicted Completion: {output.item():.2f}%")

    elif choice in boosting_model_paths:
        model_name, model_path, encoder = boosting_model_paths[choice]

        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Boosting model file not found: {model_path}")
            return

        print(f"\nLoaded boosting model: {model_name}")

        while True:
            text = input("\nEnter a sentence (or type 'exit' to quit): ").strip()
            if text.lower() == 'exit':
                print("Exiting.")
                break

            embedding = encoder(text).numpy().reshape(1, -1)
            pred = model.predict(embedding)[0] * 100
            print(f"Predicted Completion: {pred:.2f}%")

    else:
        print("Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    main()
