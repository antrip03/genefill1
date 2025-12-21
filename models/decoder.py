# models/decoder.py - Unidirectional LSTM Decoder (ORIGINAL WORKING)

import torch
import torch.nn as nn


class GapDecoder(nn.Module):
    """
    Unidirectional LSTM Decoder (4 layers, 1024 hidden).
    
    Prevents BiLSTM data leakage while maintaining high capacity.
    """
    def __init__(self, context_dim=1024, hidden_size=1024, vocab_size=4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Deep Unidirectional LSTM (4 layers)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=4,
            dropout=0.3,
            batch_first=True
        )
        
        # Context to hidden/cell initialization
        self.ctx2h = nn.Linear(context_dim, hidden_size)
        self.ctx2c = nn.Linear(context_dim, hidden_size)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, ctx, tgt_in):
        """
        Args:
            ctx: [batch, context_dim] - Encoder context
            tgt_in: [batch, seq_len] - Target input (shifted gap)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed input
        embedded = self.embed(tgt_in)
        embedded = self.dropout(embedded)
        
        # Initialize hidden states from context (for ALL 4 layers)
        h0 = self.ctx2h(ctx).unsqueeze(0).repeat(4, 1, 1)  # [4, batch, hidden_size]
        c0 = self.ctx2c(ctx).unsqueeze(0).repeat(4, 1, 1)  # [4, batch, hidden_size]
        
        # LSTM forward
        lstm_out, _ = self.lstm(embedded, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        logits = self.fc_out(lstm_out)
        
        return logits
