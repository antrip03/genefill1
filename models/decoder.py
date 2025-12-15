import torch
import torch.nn as nn

class GapDecoder(nn.Module):
    def __init__(self, context_dim=256, hidden_size=256, vocab_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.ctx2h = nn.Linear(context_dim, hidden_size)
        self.ctx2c = nn.Linear(context_dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, ctx, tgt_in):
        """
        ctx: [batch, context_dim]
        tgt_in: [batch, gap_len] (int indices)
        Returns: logits [batch, gap_len, vocab_size]
        """
        h0 = self.ctx2h(ctx).unsqueeze(0)   # [1, B, H]
        c0 = self.ctx2c(ctx).unsqueeze(0)   # [1, B, H]
        emb = self.embed(tgt_in)            # [B, G, H]
        out, _ = self.lstm(emb, (h0, c0))   # [B, G, H]
        logits = self.fc_out(out)           # [B, G, V]
        return logits

    def generate(self, ctx, max_len, start_token=0):
        """
        Greedy decoding.
        ctx: [batch, context_dim]
        Returns: [batch, max_len] indices
        """
        batch_size = ctx.size(0)
        h = self.ctx2h(ctx).unsqueeze(0)
        c = self.ctx2c(ctx).unsqueeze(0)
        
        input_token = torch.full((batch_size, 1), start_token, 
                                 dtype=torch.long, device=ctx.device)
        generated = []
        
        for _ in range(max_len):
            emb = self.embed(input_token)           # [B, 1, H]
            out, (h, c) = self.lstm(emb, (h, c))    # [B, 1, H]
            logits = self.fc_out(out)               # [B, 1, V]
            pred = logits.argmax(dim=-1)            # [B, 1]
            generated.append(pred)
            input_token = pred
        
        return torch.cat(generated, dim=1)          # [B, max_len]
