# models/encoder.py - CNN + Bi-LSTM Encoder (MODIFIED FOR YOUR TRAINED WEIGHTS)

# models/encoder.py - DEEPER CNN + Bi-LSTM Encoder

import torch
import torch.nn as nn


class CNNBiLSTMEncoder(nn.Module):
    """
    Deep CNN + Bidirectional LSTM encoder for flanking sequences.
    
    Specifications:
    - CNN: 5 layers (kernel=5, progressive channels: 64→128→256→256→256)
    - Bi-LSTM: 3 layers, 512 hidden dimensions
    - Output: context vector (1024-dim) for decoder
    """
    def __init__(self, in_channels=4, context_dim=1024):
        super().__init__()
        
        self.in_channels = in_channels
        self.context_dim = context_dim
        
        # Five Conv1d layers with progressive channel expansion
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv5 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Deep Bi-directional LSTM (3 layers, 512 hidden)
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Project bi-LSTM output (2*512 = 1024) to context_dim
        self.fc = nn.Linear(2 * 512, context_dim)

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
        
        # Transpose for LSTM: [batch, seq_len, 256]
        h = h.permute(0, 2, 1)
        
        # 3-layer Bi-LSTM
        lstm_out, (hidden, cell) = self.bilstm(h)
        
        # hidden: [6, batch, 512] (6 = 3 layers * 2 directions)
        # Extract last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]  # [batch, 512]
        backward_hidden = hidden[-1, :, :]  # [batch, 512]
        
        # Concatenate forward and backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch, 1024]
        
        # Project to context dimension
        ctx = self.fc(combined)  # [batch, 1024]
        
        return ctx
