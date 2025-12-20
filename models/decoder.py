# models/decoder.py - Bidirectional LSTM Decoder

import torch
import torch.nn as nn


class GapDecoder(nn.Module):
    """
    Bidirectional LSTM Decoder for gap filling.
    
    Instead of unidirectional decoding, uses BiLSTM to process the gap
    sequence in both directions, allowing it to look ahead while predicting.
    
    Args:
        context_dim: Dimension of encoder context vector
        hidden_size: LSTM hidden dimension
        vocab_size: Number of output classes (4 for ACGT)
    """
    def __init__(self, context_dim=512, hidden_size=512, vocab_size=4):
        super().__init__()
        
        # Embedding for input tokens (target sequence)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Bidirectional LSTM (can look both ways)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Initialize hidden/cell states from context
        self.ctx2h = nn.Linear(context_dim, hidden_size)
        self.ctx2c = nn.Linear(context_dim, hidden_size)
        
        # Output projection: BiLSTM outputs 2*hidden_size (forward+backward)
        self.fc_out = nn.Linear(2 * hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, ctx, tgt_in):
        """
        Forward pass.
        
        Args:
            ctx: [batch, context_dim] - Encoder context vector
            tgt_in: [batch, seq_len] - Target input sequence (indices)
        
        Returns:
            logits: [batch, seq_len, vocab_size] - Output probabilities
        """
        # Embed target sequence
        embedded = self.embed(tgt_in)  # [batch, seq_len, hidden_size]
        embedded = self.dropout(embedded)
        
        # Initialize h0, c0 from context (one initialization for all LSTM layers)
        # For 2-layer BiLSTM, we need [2*num_layers, batch, hidden_size]
        # = [4, batch, hidden_size] (2 layers * 2 directions)
        
        # Simple approach: use context for all layers (will get learned via gradients)
        h0_base = self.ctx2h(ctx)  # [batch, hidden_size]
        c0_base = self.ctx2c(ctx)  # [batch, hidden_size]
        
        # Replicate for 2 layers * 2 directions = 4
        h0 = h0_base.unsqueeze(0).repeat(4, 1, 1)  # [4, batch, hidden_size]
        c0 = c0_base.unsqueeze(0).repeat(4, 1, 1)  # [4, batch, hidden_size]
        
        # BiLSTM processes full sequence bidirectionally
        lstm_out, (h_n, c_n) = self.bilstm(embedded, (h0, c0))
        
        # lstm_out: [batch, seq_len, 2*hidden_size]
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary
        logits = self.fc_out(lstm_out)  # [batch, seq_len, vocab_size]
        
        return logits
