# train.py - COMPLETE CODE WITH BiLSTM ENCODER

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CNNBiLSTMEncoder, GapDecoder
from utils.dataset import GapFillDataset

# Hyperparameters
BATCH_SIZE = 32
LR = 3e-3
EPOCHS = 60
FLANK_LEN = 200
GAP_LEN = 50
CONTEXT_DIM = 256
HIDDEN_SIZE = 256
VOCAB_SIZE = 4
LSTM_HIDDEN = 128  # BiLSTM hidden size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("=" * 60)
print("Training Configuration:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Architecture: CNN + BiLSTM Encoder + 2-Layer LSTM Decoder")
print("=" * 60)
print()

# Data
with open("data/processed/gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

print(f"Total samples: {len(samples)}")
dataset = GapFillDataset(samples)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Batches per epoch: {len(train_loader)}")
print()

# Models - NEW: CNNBiLSTMEncoder with BiLSTM
encoder = CNNBiLSTMEncoder(
    in_channels=4,
    hidden_channels=128,
    lstm_hidden=LSTM_HIDDEN,
    context_dim=CONTEXT_DIM
).to(device)

decoder = GapDecoder(CONTEXT_DIM, HIDDEN_SIZE, VOCAB_SIZE).to(device)

# Count parameters
encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print(f"Encoder parameters: {encoder_params:,}")
print(f"Decoder parameters: {decoder_params:,}")
print(f"Total parameters: {encoder_params + decoder_params:,}")
print()

# Optimizer
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR,
)

criterion = nn.CrossEntropyLoss()  # No weights

print("Starting training...")
print("=" * 60)
print()

# Training loop
for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_idx, (left, right, gap) in enumerate(train_loader):
        left = left.to(device)           # [B, L, 4]
        right = right.to(device)         # [B, L, 4]
        gap = gap.to(device)             # [B, G]

        # Concatenate flanks and transpose to [B, 4, 2L]
        flanks = torch.cat([left, right], dim=1)   # [B, 2L, 4]
        flanks = flanks.permute(0, 2, 1)           # [B, 4, 2L]

        # Encode flanks with CNN + BiLSTM
        ctx = encoder(flanks)                      # [B, C]

        # Prepare decoder input with start token
        start = torch.zeros(gap.size(0), 1,
                            dtype=torch.long, device=device)
        tgt_in = torch.cat([start, gap[:, :-1]], dim=1)  # [B, G]

        # Decode to predict gap
        logits = decoder(ctx, tgt_in)              # [B, G, V]

        # Compute loss
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            gap.reshape(-1)
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)          # [B, G]
            batch_acc = (preds == gap).float().mean().item()
            total_acc += batch_acc

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1:3d}/{EPOCHS}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")

print()
print("=" * 60)
print("Training complete!")

# Save models
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
print("Saved encoder.pth and decoder.pth")
print("=" * 60)
