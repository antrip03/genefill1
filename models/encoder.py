# models/encoder.py - CNN + Bi-LSTM Encoder (3-LAYER - THE ORIGINAL WORKING ONE)

import torch
import torch.nn as nn


class CNNBiLSTMEncoder(nn.Module):
    """
    CNN + Bidirectional LSTM encoder for flanking sequences.
    
    Specifications:
    - CNN: 3 layers (kernel=5, 128 channels) - ORIGINAL WORKING ARCHITECTURE
    - Bi-LSTM: 128 hidden dimensions
    - Output: context vector (256-dim) for decoder
    """
    def __init__(self, in_channels=4, hidden_channels=128, lstm_hidden=128, context_dim=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.context_dim = context_dim
        
        # Three Conv1d layers (kernel=5) - ORIGINAL ARCHITECTURE
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Bi-directional LSTM
        # 2 layers (original spec)
        self.bilstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Project bi-LSTM output (2*lstm_hidden from bidirectional) to context_dim
        self.fc = nn.Linear(2 * lstm_hidden, context_dim)

    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: [batch, 4, seq_len] - one-hot encoded flanking sequences
        
        Returns:
            ctx: [batch, context_dim] - context vector for decoder
        """
        # Three-layer CNN feature extraction
        h = self.relu(self.conv1(x))  # [batch, 128, seq_len]
        h = self.dropout(h)
        
        h = self.relu(self.conv2(h))  # [batch, 128, seq_len]
        h = self.dropout(h)
        
        h = self.relu(self.conv3(h))  # [batch, 128, seq_len]
        
        # Transpose for LSTM: [batch, seq_len, 128]
        h = h.permute(0, 2, 1)
        
        # Bi-LSTM
        lstm_out, (hidden, cell) = self.bilstm(h)
        
        # hidden: [4, batch, lstm_hidden] (4 = 2 layers * 2 directions)
        # Extract last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]  # [batch, lstm_hidden]
        backward_hidden = hidden[-1, :, :]  # [batch, lstm_hidden]
        
        # Concatenate forward and backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch, 2*lstm_hidden]
        
        # Project to context dimension
        ctx = self.fc(combined)  # [batch, context_dim]
        
        return ctx
