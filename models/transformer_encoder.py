import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.encoding import VOCAB_SIZE, PAD_IDX, MASK_IDX  # import from encoding

class DNAMaskedEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=4, dim_ff=512, max_len=700):
        """
        max_len should cover 2*flank_len + gap_len (e.g. 300+50+300 = 650).
        """
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(
            VOCAB_SIZE, d_model, padding_idx=PAD_IDX
        )
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Predict 4 bases only
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, input_ids, pad_mask=None):
        """
        input_ids: [B, L] with MASK_IDX at masked positions
        pad_mask: [B, L] bool, True at padding positions (optional)
        """
        B, L = input_ids.shape
        device = input_ids.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B,L]

        x = self.token_emb(input_ids) + self.pos_emb(pos)               # [B,L,D]

        # PyTorch expects True where positions should be ignored
        src_key_padding_mask = pad_mask

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B,L,D]
        logits = self.classifier(h)                                     # [B,L,4]
        return logits

def masked_ce_loss(logits, targets, gap_mask):
    """
    logits: [B,L,4]; targets: [B,L]; gap_mask: [B,L] bool True at gap positions.
    """
    if gap_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    masked_logits = logits[gap_mask]   # [Nmask,4]
    masked_targets = targets[gap_mask] # [Nmask]

    return F.cross_entropy(masked_logits, masked_targets)
