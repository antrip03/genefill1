# models/encoder.py - DEEPER CNN + Bi-LSTM Encoder (Configurable)

import torch
import torch.nn as nn


class CNNBiLSTMEncoder(nn.Module):
    """
    Deep CNN + Bidirectional LSTM encoder for flanking sequences.
    
    Specifications:
    - CNN: 5 layers with progressive channels
    - Bi-LSTM: 3 layers, configurable hidden dimensions
    - Output: context vector for decoder
    """
    def __init__(self, in_channels=4, hidden_channels=128, lstm_hidden=512, context_dim=1024):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.context_dim = context_dim
        
        # Five Conv1d layers with progressive channel expansion
        # Start with hidden_channels, then double, then plateau
        c1 = hidden_channels // 2  # 64 if hidden_channels=128
        c2 = hidden_channels        # 128
        c3 = hidden_channels * 2    # 256
        c4 = hidden_channels * 2    # 256
        c5 = hidden_channels * 2    # 256
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(c3)
        
        self.conv4 = nn.Conv1d(c3, c4, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(c4)
        
        self.conv5 = nn.Conv1d(c4, c5, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(c5)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Deep Bi-directional LSTM (3 layers)
        self.bilstm = nn.LSTM(
            input_size=c5,  # Output from last CNN layer
            hidden_size=lstm_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Project bi-LSTM output (2*lstm_hidden) to context_dim
        self.fc = nn.Linear(2 * lstm_hidden, context_dim)

    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: [batch, 4, seq_len] - one-hot encoded flanking sequences
        
        Returns:
            ctx: [batch, context_dim] - context vector for decoder
        """
        # Five-layer CNN with batch norm
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.dropout(h)
        
        h = self.relu(self.bn4(self.conv4(h)))
        h = self.dropout(h)
        
        h = self.relu(self.bn5(self.conv5(h)))
        
        # Transpose for LSTM: [batch, seq_len, channels]
        h = h.permute(0, 2, 1)
        
        # 3-layer Bi-LSTM
        lstm_out, (hidden, cell) = self.bilstm(h)
        
        # hidden: [6, batch, lstm_hidden] (6 = 3 layers * 2 directions)
        # Extract last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]  # [batch, lstm_hidden]
        backward_hidden = hidden[-1, :, :]  # [batch, lstm_hidden]
        
        # Concatenate forward and backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch, 2*lstm_hidden]
        
        # Project to context dimension
        ctx = self.fc(combined)  # [batch, context_dim]
        
        return ctx
