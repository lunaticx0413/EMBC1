"""Lightweight encoders into the frozen text-embedding space and the contrastive objective."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-and-excitation channel attention over feature maps."""

    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // r), nn.GELU(), nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, ch, k=7, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.GELU(),
                                 nn.Dropout(drop), nn.Conv1d(ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class EEGEncoder(nn.Module):
    """(B, C, T) epoch -> (B, d_out). Spatial 1x1 conv, 3 temporal residual blocks, SE, projection MLP."""

    def __init__(self, n_ch, n_t, d_out, width=64, drop=0.3):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(n_ch, width, 1), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.1))
        self.temporal = nn.Sequential(ResBlock(width), nn.AvgPool1d(2), ResBlock(width), nn.AvgPool1d(2),
                                      ResBlock(width), SE(width))
        with torch.no_grad():
            t_out = self.temporal(self.spatial(torch.zeros(2, n_ch, n_t))).shape[-1]
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(width * t_out, 512), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(512, d_out))

    def forward(self, x):
        return self.head(self.temporal(self.spatial(x)))


class VectorEncoder(nn.Module):
    """(B, D) feature vector (gaze-only control) -> (B, d_out)."""

    def __init__(self, d_in, d_out, hidden=256, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def info_nce(z, type_ids, type_emb, tau):
    """EEG -> text InfoNCE over the unique target types of the batch (repeated types share one class)."""
    u, inv = torch.unique(type_ids, return_inverse=True)
    logits = F.normalize(z, dim=-1) @ type_emb[u].T / tau
    return F.cross_entropy(logits, inv)
