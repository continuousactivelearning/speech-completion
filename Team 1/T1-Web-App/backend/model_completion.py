# model_completion.py
# Text-only Regression Models for Completion Estimation

import torch
import torch.nn as nn

# ========== Model 1: MLP for Single Text Embedding (e.g., BERT) ==========
class TextRegressionNet(nn.Module):
    def __init__(self, input_dim=768):
        super(TextRegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1) * 100


# ========== Model 2: Dual Text Embedding with Residual Fusion ==========
class DualTextFusionNet(nn.Module):
    def __init__(self, input_dim1=768, input_dim2=768):
        super(DualTextFusionNet, self).__init__()
        self.text1_proj = nn.Sequential(
            nn.Linear(input_dim1, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text2_proj = nn.Sequential(
            nn.Linear(input_dim2, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, text1, text2):
        t1 = self.text1_proj(text1)
        t2 = self.text2_proj(text2)
        fused = t1 + t2 + self.fusion(t1 * t2)
        return self.regressor(fused).squeeze(1) * 100


# ========== Model 3: BiGRU for GloVe Sequence Inputs ==========
class BiGRUTextNet(nn.Module):
    def __init__(self, input_dim=300):
        super(BiGRUTextNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, h_n = self.rnn(x)  # h_n: (2, B, 256)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 512)
        return self.regressor(h_n).squeeze(1) * 100


# ========== Model 4: Attention Fusion Between Two Text Embeddings ==========
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        alpha = torch.sigmoid(self.attn(concat))
        return alpha * x1 + (1 - alpha) * x2


class DualTextAttentionNet(nn.Module):
    def __init__(self, input_dim1=384, input_dim2=768):  # e.g., SBERT + BERT
        super(DualTextAttentionNet, self).__init__()
        self.text1_proj = nn.Linear(input_dim1, 512)
        self.text2_proj = nn.Linear(input_dim2, 512)
        self.fusion = AttentionFusion(512)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, text1, text2):
        t1 = self.text1_proj(text1)
        t2 = self.text2_proj(text2)
        fused = self.fusion(t1, t2)
        return self.regressor(fused).squeeze(1) * 100


# ========== Model 5: Gated Fusion of Two Text Embeddings ==========
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, t1, t2):
        x = torch.cat([t1, t2], dim=1)
        g = self.gate(x)
        return g * t1 + (1 - g) * t2


class GatedTextFusionNet(nn.Module):
    def __init__(self, input_dim1=300, input_dim2=1024):  # e.g., GloVe + CLIP text
        super(GatedTextFusionNet, self).__init__()
        self.text1_proj = nn.Linear(input_dim1, 512)
        self.text2_proj = nn.Linear(input_dim2, 512)
        self.fusion = GatedFusion(512)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, text1, text2):
        t1 = self.text1_proj(text1)
        t2 = self.text2_proj(text2)
        fused = self.fusion(t1, t2)
        return self.regressor(fused).squeeze(1) * 100


# ========== Model 6: Simple Average Fusion for Text Embeddings ==========
class AveragedFusionNet(nn.Module):
    def __init__(self, input_dim1=300, input_dim2=768):
        super(AveragedFusionNet, self).__init__()
        self.text1_proj = nn.Linear(input_dim1, 512)
        self.text2_proj = nn.Linear(input_dim2, 512)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, text1, text2):
        t1 = self.text1_proj(text1)
        t2 = self.text2_proj(text2)
        fused = (t1 + t2) / 2
        return self.regressor(fused).squeeze(1) * 100


# ========== Model Preview ==========
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")

    print("\n=== TextRegressionNet ===")
    print(TextRegressionNet().to(device))

    print("\n=== DualTextFusionNet ===")
    print(DualTextFusionNet().to(device))

    print("\n=== BiGRUTextNet ===")
    print(BiGRUTextNet().to(device))

    print("\n=== DualTextAttentionNet ===")
    print(DualTextAttentionNet().to(device))

    print("\n=== GatedTextFusionNet ===")
    print(GatedTextFusionNet().to(device))

    print("\n=== AveragedFusionNet ===")
    print(AveragedFusionNet().to(device))
