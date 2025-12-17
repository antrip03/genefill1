# models/encoder.py - NEW CNN + BiLSTM Encoder

import torch
import torch.nn as nn


class CNNBiLSTMEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=128, lstm_hidden=128, context_dim=256):
        super().__init__()
        
        # Three Conv1d layers for local feature extraction
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Bidirectional LSTM to process CNN features
        self.bilstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,  # KEY: bidirectional=True
            dropout=0.2
        )
        
        # Final linear layer: BiLSTM outputs 2*lstm_hidden (forward + backward)
        self.fc = nn.Linear(2 * lstm_hidden, context_dim)

    def forward(self, x):
        """
        x: [batch, 4, seq_len] (one-hot encoded DNA)
        Returns: [batch, context_dim] context vector
        """
        # CNN feature extraction
        h = self.relu(self.conv1(x))           # [B, H, L]
        h = self.dropout(h)
        h = self.relu(self.conv2(h))           # [B, H, L]
        h = self.dropout(h)
        h = self.relu(self.conv3(h))           # [B, H, L]
        
        # Prepare for LSTM: [B, H, L] → [B, L, H]
        h = h.permute(0, 2, 1)                 # [B, L, H]
        
        # BiLSTM processes sequence in both directions
        lstm_out, (hidden, cell) = self.bilstm(h)  # lstm_out: [B, L, 2*lstm_hidden]
        
        # Take the last time step from both directions
        # hidden: [4, B, lstm_hidden] (4 = 2 layers * 2 directions)
        # Extract last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]      # [B, lstm_hidden]
        backward_hidden = hidden[-1, :, :]     # [B, lstm_hidden]
        
        # Concatenate forward and backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, 2*lstm_hidden]
        
        # Project to context dimension
        ctx = self.fc(combined)                # [B, context_dim]
        
        return ctx


