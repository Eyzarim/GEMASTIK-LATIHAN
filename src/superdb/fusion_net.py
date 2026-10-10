"""Jaringan kecil di atas embedding pretrained beku; kelas top-level bisa disimpan."""
import torch
from torch import nn


class GatedFusionNet(nn.Module):
    def __init__(self, dimensions, out_dim, hidden=64):
        super().__init__()
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2))
                                     for d in dimensions])
        self.gates = nn.ModuleList([nn.Linear(hidden, 1) for _ in dimensions])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, blocks):
        h = [tower(x) for tower, x in zip(self.towers, blocks)]
        weights = torch.cat([gate(x) for gate, x in zip(self.gates, h)], dim=1).softmax(1)
        return self.head((torch.stack(h, dim=1) * weights.unsqueeze(-1)).sum(1))
