# models/decoder.py - WORKING DECODER (Your Old Architecture, 512-dim)

import torch
import torch.nn as nn


class GapDecoder(nn.Module):
    """
    LSTM Decoder with context-initialized hidden AND cell states.
    This is your PROVEN working architecture.
    """
    def __init__(self, context_dim=512, hidden_size=512, vocab_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=2, 
            dropout=0.2, 
            batch_first=True
        )
        
        # Context to hidden state (h0)
        self.ctx2h = nn.Linear(context_dim, hidden_size)
        
        # Context to cell state (c0) - THIS WAS MISSING!
        self.ctx2c = nn.Linear(context_dim, hidden_size)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, ctx, tgt_in):
        """
        Forward pass with context-initialized h0 AND c0.
        
        Args:
            ctx: [batch, context_dim]
            tgt_in: [batch, gap_len] (int indices)
        
        Returns:
            logits: [batch, gap_len, vocab_size]
        """
        # Initialize BOTH hidden and cell states from context
        h0 = self.ctx2h(ctx).unsqueeze(0).repeat(2, 1, 1)  # [2, B, H]
        c0 = self.ctx2c(ctx).unsqueeze(0).repeat(2, 1, 1)  # [2, B, H]
        
        # Embed target tokens
        emb = self.embed(tgt_in)  # [B, G, H]
        
        # LSTM forward
        out, _ = self.lstm(emb, (h0, c0))  # [B, G, H]
        
        # Output projection
        logits = self.fc_out(out)  # [B, G, V]
        
        return logits

    def generate(self, ctx, max_len, start_token=0):
        """
        Greedy decoding.
        """
        batch_size = ctx.size(0)
        
        # Initialize hidden and cell states from context
        h = self.ctx2h(ctx).unsqueeze(0).repeat(2, 1, 1)  # [2, B, H]
        c = self.ctx2c(ctx).unsqueeze(0).repeat(2, 1, 1)  # [2, B, H]
        
        input_token = torch.full((batch_size, 1), start_token,
                                 dtype=torch.long, device=ctx.device)
        generated = []
        
        for _ in range(max_len):
            emb = self.embed(input_token)  # [B, 1, H]
            out, (h, c) = self.lstm(emb, (h, c))  # [B, 1, H]
            logits = self.fc_out(out)  # [B, 1, V]
            pred = logits.argmax(dim=-1)  # [B, 1]
            generated.append(pred)
            input_token = pred
        
        return torch.cat(generated, dim=1)  # [B, max_len]
