import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CNNEncoder, GapDecoder
from utils.dataset import GapFillDataset

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 60
FLANK_LEN = 200
GAP_LEN = 50
CONTEXT_DIM = 256
HIDDEN_SIZE = 256
VOCAB_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Data
with open("data/processed/gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

dataset = GapFillDataset(samples)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
encoder = CNNEncoder(4, 128, CONTEXT_DIM).to(device)
decoder = GapDecoder(CONTEXT_DIM, HIDDEN_SIZE, VOCAB_SIZE).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR,
)

wA = 1.0 / 0.345387
wC = 1.0 / 0.157991
wG = 1.0 / 0.159173
wT = 1.0 / 0.337448
weights = torch.tensor([wA, wC, wG, wT], device=device)
criterion = nn.CrossEntropyLoss(weight=weights)



for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_acc = 0.0
    for left, right, gap in train_loader:
        left = left.to(device)           # [B, L, 4]
        right = right.to(device)         # [B, L, 4]
        gap = gap.to(device)             # [B, G]

        flanks = torch.cat([left, right], dim=1)   # [B, 2L, 4]
        flanks = flanks.permute(0, 2, 1)           # [B, 4, 2L]

        ctx = encoder(flanks)                      # [B, C]

        start = torch.zeros(gap.size(0), 1,
                            dtype=torch.long, device=device)
        tgt_in = torch.cat([start, gap[:, :-1]], dim=1)  # [B, G]

        logits = decoder(ctx, tgt_in)              # [B, G, V]

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            gap.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)          # [B, G]
            batch_acc = (preds == gap).float().mean().item()
            total_acc += batch_acc

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")

torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
print("Saved encoder.pth and decoder.pth")
