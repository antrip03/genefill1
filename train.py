# train.py - MULTI-GPU TRAINING WITH BiLSTM ENCODER

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CNNBiLSTMEncoder, GapDecoder
from utils.dataset import GapFillDataset

# =============================================================================
# GPU Setup
# =============================================================================

print("="*70)
print("GPU DETECTION")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

# =============================================================================
# Hyperparameters
# =============================================================================

BATCH_SIZE = 64  # Increased from 32 for multi-GPU
LR = 3e-3
EPOCHS = 60
FLANK_LEN = 200
GAP_LEN = 50
CONTEXT_DIM = 256
HIDDEN_SIZE = 256
VOCAB_SIZE = 4
LSTM_HIDDEN = 128

# Multi-GPU settings
USE_MULTI_GPU = True  # Set False to force single GPU
SAVE_EVERY = 10  # Save checkpoint every N epochs

# Device setup
if not torch.cuda.is_available():
    device = torch.device("cpu")
    use_multi_gpu = False
    print("⚠️  CUDA not available, using CPU")
else:
    device = torch.device("cuda:0")
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = USE_MULTI_GPU and num_gpus > 1
    
    if use_multi_gpu:
        print(f"✓ Multi-GPU enabled: Using {num_gpus} GPUs")
        BATCH_SIZE = BATCH_SIZE * num_gpus  # Scale batch size
        print(f"  Scaled batch size: {BATCH_SIZE}")
    else:
        print(f"✓ Single GPU mode: {torch.cuda.get_device_name(0)}")

print()
print("="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Architecture: CNN + BiLSTM Encoder + 2-Layer LSTM Decoder")
print(f"  Multi-GPU: {'Yes' if use_multi_gpu else 'No'}")
print("="*70)
print()

# =============================================================================
# Data Loading
# =============================================================================

with open("data/processed/mixed_gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

print(f"Total samples: {len(samples):,}")
dataset = GapFillDataset(samples)

# Multi-GPU: use multiple workers and pin memory
num_workers = 2 if use_multi_gpu else 0
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

print(f"Batches per epoch: {len(train_loader):,}")
print(f"Workers: {num_workers}")
print()

# =============================================================================
# Models
# =============================================================================

print("Initializing models...")

encoder = CNNBiLSTMEncoder(
    in_channels=4,
    hidden_channels=128,
    lstm_hidden=LSTM_HIDDEN,
    context_dim=CONTEXT_DIM
)

decoder = GapDecoder(CONTEXT_DIM, HIDDEN_SIZE, VOCAB_SIZE)

# Wrap with DataParallel for multi-GPU
if use_multi_gpu:
    print(f"Wrapping models with DataParallel ({torch.cuda.device_count()} GPUs)")
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

encoder.to(device)
decoder.to(device)

# Count parameters
encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
total_params = encoder_params + decoder_params

print(f"Encoder parameters: {encoder_params:,}")
print(f"Decoder parameters: {decoder_params:,}")
print(f"Total parameters: {total_params:,}")
print()

# =============================================================================
# Optimizer & Scheduler
# =============================================================================

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True
)

print("Optimizer: AdamW")
print(f"Learning rate: {LR}")
print("Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
print()

# =============================================================================
# Training Loop
# =============================================================================

print("="*70)
print("TRAINING START")
print("="*70)
print()

best_loss = float('inf')
best_acc = 0.0

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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 
            max_norm=1.0
        )
        
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)          # [B, G]
            batch_acc = (preds == gap).float().mean().item()
            total_acc += batch_acc
        
        # Progress logging (every 50 batches)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}: "
                  f"loss={loss.item():.4f}, acc={batch_acc:.3f}")

    # Epoch statistics
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    
    print(f"\nEpoch {epoch+1:3d}/{EPOCHS}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")
    
    # Step scheduler
    scheduler.step(avg_loss)
    
    # Save checkpoints
    is_best = False
    if avg_loss < best_loss:
        best_loss = avg_loss
        is_best = True
    
    if avg_acc > best_acc:
        best_acc = avg_acc
    
    # Save every N epochs or if best model
    if (epoch + 1) % SAVE_EVERY == 0 or is_best:
        print(f"  Saving checkpoint... ", end='')
        
        # For DataParallel, need to save module.state_dict()
        encoder_state = encoder.module.state_dict() if use_multi_gpu else encoder.state_dict()
        decoder_state = decoder.module.state_dict() if use_multi_gpu else decoder.state_dict()
        
        torch.save(encoder_state, 'encoder.pth')
        torch.save(decoder_state, 'decoder.pth')
        
        if is_best:
            print(f"✓ New best model! (loss: {best_loss:.4f})")
            torch.save(encoder_state, 'encoder_best.pth')
            torch.save(decoder_state, 'decoder_best.pth')
        else:
            print("✓")
    
    print()

# =============================================================================
# Training Complete
# =============================================================================

print("="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best loss: {best_loss:.4f}")
print(f"Best accuracy: {best_acc:.3f}")
print()
print("Saved models:")
print("  encoder.pth, decoder.pth (latest)")
print("  encoder_best.pth, decoder_best.pth (best loss)")
print("="*70)

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("\n✓ GPU memory cleared")

