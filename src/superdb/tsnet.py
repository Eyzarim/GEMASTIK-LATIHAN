"""Jaringan PyTorch untuk time series, di modul sendiri agar model terlatih bisa di-pickle dan dimuat di proses lain."""
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.r = nn.LSTM(1, hidden, batch_first=True)
        self.o = nn.Linear(hidden, n_classes)

    def forward(self, x):                      # x: (batch, 1, T)
        o, _ = self.r(x.transpose(1, 2))
        return self.o(o[:, -1])
