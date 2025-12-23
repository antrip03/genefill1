# train_mlm.py - BERT-style masked DNA training (E. coli, multi-GPU)

import pickle
import time
import torch
from torch.utils.data import DataLoader

from models import DNAMaskedEncoder
from utils.masked_dataset import MaskedGapDataset
from models.transformer_encoder import masked_ce_loss
from utils.encoding import PAD_IDX

print("=" * 70)
print("BERT-STYLE MASKED DNA TRAINING (ENCODER ONLY)")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print("=" * 70)
print()

# ---------------------------------------------------------------------
# Device + multi-GPU
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_MULTI_GPU = True

if DEVICE == "cuda":
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = USE_MULTI_GPU and num_gpus > 1
    if use_multi_gpu:
        print(f"✓ Multi-GPU enabled: {num_gpus} GPUs")
    else:
        print("✓ Single GPU mode")
else:
    use_multi_gpu = False
    print("⚠️ Using CPU")

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 70

DATA_PATH = "data/processed/ecoli_gapfill_easy_20k_100_10.pkl"
SAVE_PATH = "dna_mlm_transformer.pth"

print()
print("CONFIG")
print("=" * 70)
print(f"  Dataset file: {DATA_PATH}")
print(f"  Batch size:   {BATCH_SIZE}")
print(f"  Learning rate:{LR}")
print(f"  Epochs:       {EPOCHS}")
print(f"  Device:       {DEVICE}")
print(f"  Multi-GPU:    {'Yes' if use_multi_gpu else 'No'}")
print("=" * 70)
print()

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
print("Loading samples...")
with open(DATA_PATH, "rb") as f:
    samples = pickle.load(f)

print(f"Total samples: {len(samples):,}")

dataset = MaskedGapDataset(samples)

def collate_fn(batch):
    # batch: list of (masked, target, gap_mask), each [L]
    masked_list, target_list, mask_list = zip(*batch)
    lengths = [x.shape[0] for x in masked_list]
    max_len = max(lengths)

    padded_masked, padded_target, padded_gapmask = [], [], []

    for masked, target, gmask in zip(masked_list, target_list, mask_list):
        pad_len = max_len - masked.shape[0]
        if pad_len > 0:
            pad_val = PAD_IDX
            masked = torch.cat(
                [masked, torch.full((pad_len,), pad_val, dtype=torch.long)],
                dim=0
            )
            target = torch.cat(
                [target, torch.full((pad_len,), pad_val, dtype=torch.long)],
                dim=0
            )
            gmask = torch.cat(
                [gmask, torch.zeros(pad_len, dtype=torch.bool)],
                dim=0
            )
        padded_masked.append(masked)
        padded_target.append(target)
        padded_gapmask.append(gmask)

    return (
        torch.stack(padded_masked, dim=0),   # [B,L]
        torch.stack(padded_target, dim=0),   # [B,L]
        torch.stack(padded_gapmask, dim=0),  # [B,L]
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

print(f"Batches per epoch: {len(loader)}")
print()

# ---------------------------------------------------------------------
# Model / Optimizer
# ---------------------------------------------------------------------
model = DNAMaskedEncoder()

if use_multi_gpu:
    import torch.nn as nn
    print("Wrapping model with DataParallel")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Model initialized: DNAMaskedEncoder")
print()

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
print("=" * 70)
print("TRAINING START (MASKED LM)")
print("=" * 70)

for epoch in range(EPOCHS):
    model.train()
    epoch_start = time.time()

    total_loss = 0.0
    total_mask_tokens = 0
    total_correct_tokens = 0

    for batch_idx, (masked_ids, targets, gap_mask) in enumerate(loader):
        masked_ids = masked_ids.to(DEVICE)
        targets = targets.to(DEVICE)
        gap_mask = gap_mask.to(DEVICE)

        pad_mask = (masked_ids == PAD_IDX)  # True where padding

        logits = model(masked_ids, pad_mask)        # [B,L,4]
        loss = masked_ce_loss(logits, targets, gap_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ----- masked-token stats -----
        with torch.no_grad():
            preds = logits.argmax(dim=-1)           # [B,L]
            correct = ((preds == targets) & gap_mask).sum().item()
            n_mask = gap_mask.sum().item()
            if n_mask == 0:
                n_mask = 1  # avoid division by zero

        total_loss += loss.item() * n_mask
        total_mask_tokens += n_mask
        total_correct_tokens += correct

        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / total_mask_tokens
            avg_acc = total_correct_tokens / total_mask_tokens
            print(
                f"  Epoch {epoch+1:03d} | "
                f"Batch {batch_idx+1:04d}/{len(loader)} | "
                f"Masked-token loss: {avg_loss:.4f} | "
                f"Masked-token acc:  {avg_acc:.4f}"
            )

    avg_loss = total_loss / max(total_mask_tokens, 1)
    avg_acc = total_correct_tokens / max(total_mask_tokens, 1)
    epoch_time = time.time() - epoch_start
    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Masked-token loss: {avg_loss:.4f} | "
        f"Masked-token acc:  {avg_acc:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )
    print("-" * 70)

core_model = model.module if use_multi_gpu else model
torch.save(core_model.state_dict(), SAVE_PATH)
print(f"\n✓ Saved masked-LM model to {SAVE_PATH}")

