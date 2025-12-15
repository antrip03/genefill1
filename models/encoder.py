import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=128, context_dim=256):
        super().__init__()
        # Three Conv1d layers, same channels and kernel size
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_channels, context_dim)

    def forward(self, x):
        # x: [batch, 4, seq_len]
        h = self.relu(self.conv1(x))   # [B, H, L]
        h = self.relu(self.conv2(h))   # [B, H, L]
        h = self.relu(self.conv3(h))   # [B, H, L]
        h = self.pool(h)               # [B, H, 1]
        h = h.squeeze(-1)              # [B, H]
        ctx = self.fc(h)               # [B, context_dim]
        return ctx

