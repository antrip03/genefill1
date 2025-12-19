# models/decoder.py - LSTM Decoder (with attention)
# Increased hidden size to 512

import torch
import torch.nn as nn


class GapDecoder(nn.Module):
    """
    LSTM Decoder with context attention at every step.
    
    Specifications:
    - Hidden size: 512 (matching DLGapCloser)
    - Embedding dim: 512
    - 2 LSTM layers with dropout
    - Context concatenation at every decoding step
    """
    def __init__(self, context_dim=512, hidden_size=512, vocab_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.context_dim = context_dim
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM: concatenates embedding + context at every step
        self.lstm = nn.LSTM(
            input_size=hidden_size + context_dim,  # embedding + context
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, ctx, tgt_in):
        """
        Forward pass with teacher forcing.
        
        Args:
            ctx: [batch, context_dim] - encoder context
            tgt_in: [batch, gap_len] - target indices
        
        Returns:
            logits: [batch, gap_len, vocab_size]
        """
        batch_size, gap_len = tgt_in.shape
        
        # Embed target tokens
        emb = self.embed(tgt_in)  # [B, G, hidden_size]
        emb = self.dropout(emb)
        
        # Expand context to match sequence length (KEY: at EVERY step)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, gap_len, -1)  # [B, G, context_dim]
        
        # Concatenate embedding with context
        lstm_input = torch.cat([emb, ctx_expanded], dim=-1)  # [B, G, hidden_size+context_dim]
        
        # LSTM processing
        out, _ = self.lstm(lstm_input)  # [B, G, hidden_size]
        out = self.dropout(out)
        
        # Output projection
        logits = self.fc_out(out)  # [B, G, vocab_size]
        
        return logits

    def generate(self, ctx, max_len, start_token=0):
        """
        Autoregressive generation with context attention.
        
        Args:
            ctx: [batch, context_dim]
            max_len: gap length to generate
            start_token: starting token index
        
        Returns:
            generated: [batch, max_len]
        """
        batch_size = ctx.size(0)
        device = ctx.device
        
        input_token = torch.full((batch_size, 1), start_token, 
                                 dtype=torch.long, device=device)
        generated = []
        hidden = None
        
        for step in range(max_len):
            # Embed current token
            emb = self.embed(input_token)  # [B, 1, hidden_size]
            
            # Concatenate with context (KEY: at EVERY step)
            ctx_expanded = ctx.unsqueeze(1)  # [B, 1, context_dim]
            lstm_input = torch.cat([emb, ctx_expanded], dim=-1)
            
            # LSTM step
            out, hidden = self.lstm(lstm_input, hidden)  # [B, 1, hidden_size]
            
            # Predict next token
            logits = self.fc_out(out)  # [B, 1, vocab_size]
            pred = logits.argmax(dim=-1)  # [B, 1]
            
            generated.append(pred)
            input_token = pred
        
        return torch.cat(generated, dim=1)  # [B, max_len]
