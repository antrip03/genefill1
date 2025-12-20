# models/encoder.py - CNN + Bi-LSTM Encoder (256-dim REVERTED)
# Aligned with proven working specifications

import torch
import torch.nn as nn


class CNNBiLSTMEncoder(nn.Module):
    """
    CNN + Bidirectional LSTM encoder for flanking sequences.
    
    Specifications:
    - CNN: 1D conv with kernel=3, output=128 channels
    - Bi-LSTM: 128 hidden dimensions
    - Output: context vector (256-dim) for decoder
    """
    def __init__(self, in_channels=4, hidden_channels=128, lstm_hidden=128, context_dim=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.context_dim = context_dim
        
        # CNN feature extraction (1D convolution)
        # Kernel size=3, output channels=128
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1
        )
        
        # Activation
        self.relu = nn.ReLU()
        
        # Max pooling to reduce dimensionality
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bi-directional LSTM
        # 128 hidden dimensions (proven working)
        self.bilstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Project bi-LSTM output (2*lstm_hidden from bidirectional) to context_dim
        self.fc_context = nn.Linear(2 * lstm_hidden, context_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: [batch, 4, seq_len] - one-hot encoded flanking sequences
        
        Returns:
            ctx: [batch, context_dim] - context vector for decoder
        """
        # CNN feature extraction
        # x: [batch, 4, seq_len]
        conv_out = self.conv1d(x)  # [batch, 128, seq_len]
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        
        # Max pooling
        pooled = self.pool(conv_out)  # [batch, 128, seq_len/2]
        
        # Transpose for LSTM: [batch, seq_len/2, 128]
        lstm_input = pooled.transpose(1, 2)
        
        # Bi-LSTM
        lstm_out, (h_n, c_n) = self.bilstm(lstm_input)
        
        # lstm_out: [batch, seq_len/2, 2*lstm_hidden]
        # h_n: [2, batch, lstm_hidden] (2 for bidirectional)
        
        # Concatenate final hidden states from both directions
        h_final = torch.cat([h_n[0], h_n[1]], dim=1)  # [batch, 2*lstm_hidden=256]
        
        # Project to context dimension
        ctx = self.fc_context(h_final)  # [batch, context_dim=256]
        ctx = self.dropout(ctx)
        
        return ctx
