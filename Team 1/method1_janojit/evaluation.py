# evaluation.py
# Evaluate all models using appropriate text embeddings

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from model_completion import (
    TextRegressionNet,
    DualTextFusionNet,
    BiGRUTextNet,
    DualTextAttentionNet,
    GatedTextFusionNet,
    AveragedFusionNet
)

# Ensure necessary folders exist
os.makedirs("model_visualisation", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training config
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 1.5  # Weight scaling exponent

# Load preprocessed data helper
def load_data(paths, text_keys):
    if isinstance(paths, str):
        data = np.load(paths)
        tensors = [torch.tensor(data[k], dtype=torch.float32) for k in text_keys]
        labels = torch.tensor(data['labels'], dtype=torch.float32)
    else:  # List of paths for multi-input
        tensors = []
        labels = None
        for path in paths:
            data = np.load(path)
            tensors.append(torch.tensor(data['text'], dtype=torch.float32))
            if labels is None:
                labels = torch.tensor(data['labels'], dtype=torch.float32)
    return tensors, labels

# Train/eval loop
def train_and_evaluate(model, train_loader, val_loader, test_loader, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for *inputs, y in train_loader:
            inputs = [i.to(device) for i in inputs]
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(*inputs)

            # Weighted MSE loss
            weight = (y / 100).clamp(min=1e-4) ** GAMMA
            loss = (weight * (preds - y) ** 2).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for *inputs, y in val_loader:
                inputs = [i.to(device) for i in inputs]
                y = y.to(device)
                preds = model(*inputs)
                weight = (y / 100).clamp(min=1e-4) ** GAMMA
                loss = (weight * (preds - y) ** 2).mean()
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

        print(f"[{model_name}] Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE Loss")
    plt.title(f"Loss Curve: {model_name}")
    plt.legend()
    plt.savefig(f"model_visualisation/{model_name}_loss.jpg")
    plt.close()

    # Final evaluation on train and test
    model.eval()
    def compute_r2(loader):
        y_true, y_pred = [], []
        with torch.no_grad():
            for *inputs, y in loader:
                inputs = [i.to(device) for i in inputs]
                y = y.to(device)
                preds = model(*inputs)
                y_true.append(y.cpu().numpy())
                y_pred.append(preds.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return r2_score(y_true, y_pred)

    train_r2 = compute_r2(train_loader)
    test_r2 = compute_r2(test_loader)

    # Save model
    torch.save(model.state_dict(), f"models/{model_name}.pt")

    return train_losses[-1], val_losses[-1], train_r2, test_r2

# Model configurations
model_configs = [
    {
        "name": "TextRegressionNet_BERT",
        "model": TextRegressionNet(768),
        "preprocessed": ["preprocessed/preprocessed1_train.npz",
                         "preprocessed/preprocessed1_val.npz",
                         "preprocessed/preprocessed1_test.npz"],
        "keys": ["text"]
    },
    {
        "name": "TextRegressionNet_DistilBERT",
        "model": TextRegressionNet(768),
        "preprocessed": ["preprocessed/preprocessed2_train.npz",
                         "preprocessed/preprocessed2_val.npz",
                         "preprocessed/preprocessed2_test.npz"],
        "keys": ["text"]
    },
    {
        "name": "BiGRUTextNet_GloVe",
        "model": BiGRUTextNet(300),
        "preprocessed": ["preprocessed/preprocessed3_train.npz",
                         "preprocessed/preprocessed3_val.npz",
                         "preprocessed/preprocessed3_test.npz"],
        "keys": ["text"]
    },
    {
        "name": "TextRegressionNet_SBERT",
        "model": TextRegressionNet(384),
        "preprocessed": ["preprocessed/preprocessed4_train.npz",
                         "preprocessed/preprocessed4_val.npz",
                         "preprocessed/preprocessed4_test.npz"],
        "keys": ["text"]
    },
    {
        "name": "TextRegressionNet_RoBERTa",
        "model": TextRegressionNet(768),
        "preprocessed": ["preprocessed/preprocessed5_train.npz",
                         "preprocessed/preprocessed5_val.npz",
                         "preprocessed/preprocessed5_test.npz"],
        "keys": ["text"]
    },
    {
        "name": "DualTextFusionNet_BERT_RoBERTa",
        "model": DualTextFusionNet(768, 768),
        "preprocessed": [
            ["preprocessed/preprocessed1_train.npz", "preprocessed/preprocessed5_train.npz"],
            ["preprocessed/preprocessed1_val.npz", "preprocessed/preprocessed5_val.npz"],
            ["preprocessed/preprocessed1_test.npz", "preprocessed/preprocessed5_test.npz"]
        ],
        "keys": ["text", "text"]
    },
    {
        "name": "DualTextAttentionNet_SBERT_BERT",
        "model": DualTextAttentionNet(384, 768),
        "preprocessed": [
            ["preprocessed/preprocessed4_train.npz", "preprocessed/preprocessed1_train.npz"],
            ["preprocessed/preprocessed4_val.npz", "preprocessed/preprocessed1_val.npz"],
            ["preprocessed/preprocessed4_test.npz", "preprocessed/preprocessed1_test.npz"]
        ],
        "keys": ["text", "text"]
    },
    {
        "name": "GatedTextFusionNet_GloVe_RoBERTa",
        "model": GatedTextFusionNet(300, 768),
        "preprocessed": [
            ["preprocessed/preprocessed3_train.npz", "preprocessed/preprocessed5_train.npz"],
            ["preprocessed/preprocessed3_val.npz", "preprocessed/preprocessed5_val.npz"],
            ["preprocessed/preprocessed3_test.npz", "preprocessed/preprocessed5_test.npz"]
        ],
        "keys": ["text", "text"]
    },
    {
        "name": "AveragedFusionNet_GloVe_BERT",
        "model": AveragedFusionNet(300, 768),
        "preprocessed": [
            ["preprocessed/preprocessed3_train.npz", "preprocessed/preprocessed1_train.npz"],
            ["preprocessed/preprocessed3_val.npz", "preprocessed/preprocessed1_val.npz"],
            ["preprocessed/preprocessed3_test.npz", "preprocessed/preprocessed1_test.npz"]
        ],
        "keys": ["text", "text"]
    }
]

# Evaluate all models
results = []
for config in model_configs:
    print(f"\n===== Evaluating {config['name']} =====")
    model = config['model'].to(device)
    train_inputs, train_labels = load_data(config['preprocessed'][0], config['keys'])
    val_inputs, val_labels = load_data(config['preprocessed'][1], config['keys'])
    test_inputs, test_labels = load_data(config['preprocessed'][2], config['keys'])

    train_ds = TensorDataset(*train_inputs, train_labels)
    val_ds = TensorDataset(*val_inputs, val_labels)
    test_ds = TensorDataset(*test_inputs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    train_loss, val_loss, train_r2, test_r2 = train_and_evaluate(
        model, train_loader, val_loader, test_loader, config['name'])

    results.append({
        "model": config['name'],
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("model_visualisation/result_df.csv", index=False)

# Plot side-by-side bar chart for Train vs Test R²
plt.figure(figsize=(10, 6))
x = range(len(df))
plt.bar(x, df["train_r2"], width=0.4, label='Train R2', align='center')
plt.bar([i + 0.4 for i in x], df["test_r2"], width=0.4, label='Test R2', align='center')
plt.xticks([i + 0.2 for i in x], df["model"], rotation=45, ha='right')
plt.ylabel("R² Score")
plt.title("Train vs Test R² Score Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("model_visualisation/r2_comparison.png")
plt.close()

print("\nAll models trained, saved, and visualized successfully.")
