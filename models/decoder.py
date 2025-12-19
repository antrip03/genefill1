# models/decoder.py - Decoder with Context Gating

import torch
import torch.nn as nn


class GapDecoder(nn.Module):
    """
    LSTM Decoder with context gating mechanism.
    Forces the decoder to use context through a learnable gate.
    """
    def __init__(self, context_dim=512, hidden_size=512, vocab_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.context_dim = context_dim
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM (takes only embedding, not context)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Context gating mechanism (NEW!)
        # Gates control how much context vs LSTM state to use
        self.context_gate = nn.Linear(context_dim + hidden_size, hidden_size)
        self.context_transform = nn.Linear(context_dim, hidden_size)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, ctx, tgt_in):
        """
        Forward pass with context gating.
        
        Args:
            ctx: [batch, context_dim]
            tgt_in: [batch, gap_len]
        
        Returns:
            logits: [batch, gap_len, vocab_size]
        """
        batch_size, gap_len = tgt_in.shape
        
        # Embed target tokens
        emb = self.embed(tgt_in)  # [B, G, hidden_size]
        emb = self.dropout(emb)
        
        # LSTM processing (without context)
        lstm_out, _ = self.lstm(emb)  # [B, G, hidden_size]
        lstm_out = self.dropout(lstm_out)
        
        # Transform context to same dimension as LSTM output
        ctx_transformed = self.context_transform(ctx)  # [B, hidden_size]
        ctx_expanded = ctx_transformed.unsqueeze(1).expand(-1, gap_len, -1)  # [B, G, hidden_size]
        
        # Compute gate (determines context vs LSTM balance)
        gate_input = torch.cat([lstm_out, ctx_expanded], dim=-1)  # [B, G, 2*hidden_size]
        gate = torch.sigmoid(self.context_gate(gate_input))  # [B, G, hidden_size]
        
        # Apply gating: interpolate between LSTM and context
        gated_out = gate * ctx_expanded + (1 - gate) * lstm_out  # [B, G, hidden_size]
        
        # Output projection
        logits = self.fc_out(gated_out)  # [B, G, vocab_size]
        
        return logits

    def generate(self, ctx, max_len, start_token=0):
        """
        Autoregressive generation with context gating.
        """
        batch_size = ctx.size(0)
        device = ctx.device
        
        input_token = torch.full((batch_size, 1), start_token, 
                                 dtype=torch.long, device=device)
        generated = []
        hidden = None
        
        # Transform context once
        ctx_transformed = self.context_transform(ctx)  # [B, hidden_size]
        
        for step in range(max_len):
            # Embed current token
            emb = self.embed(input_token)  # [B, 1, hidden_size]
            
            # LSTM step
            lstm_out, hidden = self.lstm(emb, hidden)  # [B, 1, hidden_size]
            
            # Apply context gating
            ctx_expanded = ctx_transformed.unsqueeze(1)  # [B, 1, hidden_size]
            gate_input = torch.cat([lstm_out, ctx_expanded], dim=-1)
            gate = torch.sigmoid(self.context_gate(gate_input))
            gated_out = gate * ctx_expanded + (1 - gate) * lstm_out
            
            # Predict next token
            logits = self.fc_out(gated_out)  # [B, 1, vocab_size]
            pred = logits.argmax(dim=-1)  # [B, 1]
            
            generated.append(pred)
            input_token = pred
        
        return torch.cat(generated, dim=1)  # [B, max_len]
