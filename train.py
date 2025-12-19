# train.py - MULTI-GPU TRAINING WITH 512-DIM BiLSTM (DLGapCloser Specs)

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
# Hyperparameters (DLGapCloser Specs + 1e-3 LR)
# =============================================================================

BATCH_SIZE = 64           # Paper spec
LR = 1e-3                 # YOUR SPEC (decreased from 3e-3)
EPOCHS = 1500             # Paper spec (with early stopping)
FLANK_LEN = 200
GAP_LEN = 50
CONTEXT_DIM = 512         # INCREASED (was 256)
HIDDEN_SIZE = 512         # INCREASED (was 256)
VOCAB_SIZE = 4
LSTM_HIDDEN = 512         # INCREASED (was 128) - Paper spec
EARLY_STOPPING_PATIENCE = 50

# Multi-GPU settings
USE_MULTI_GPU = True      # Set False to force single GPU
SAVE_EVERY = 10           # Save checkpoint every N epochs

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
print("TRAINING CONFIGURATION (DLGapCloser Aligned)")
print("="*70)
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"  Context Dim: {CONTEXT_DIM}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  LSTM Hidden: {LSTM_HIDDEN}")
print(f"  Architecture: CNN(32ch) + BiLSTM({LSTM_HIDDEN}dim) Encoder")
print(f"                + 2-Layer LSTM({HIDDEN_SIZE}dim) Decoder")
print(f"  Multi-GPU: {'Yes' if use_multi_gpu else 'No'}")
print("="*70)
print()

# =============================================================================
# Data Loading
# =============================================================================

with open("data/processed/mixed_gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

print(f"Total samples: {len(samples):,}")

# Split: 1000 for test, rest for training
train_samples = samples[:-1000]
test_samples = samples[-1000:]

print(f"Train samples: {len(train_samples):,}")
print(f"Test samples: {len(test_samples):,}")

train_dataset = GapFillDataset(train_samples)
test_dataset = GapFillDataset(test_samples)

# Multi-GPU: use multiple workers and pin memory
num_workers = 4 if use_multi_gpu else 0
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

print(f"Train batches per epoch: {len(train_loader):,}")
print(f"Test batches per epoch: {len(test_loader):,}")
print(f"Workers: {num_workers}")
print()

# =============================================================================
# Models (512-dim Specs)
# =============================================================================

print("Initializing models (512-dim)...")

encoder = CNNBiLSTMEncoder(
    in_channels=4,
    hidden_channels=32,           # Paper spec
    lstm_hidden=LSTM_HIDDEN,      # 512 (NEW)
    context_dim=CONTEXT_DIM       # 512 (NEW)
)

decoder = GapDecoder(
    context_dim=CONTEXT_DIM,      # 512 (NEW)
    hidden_size=HIDDEN_SIZE,      # 512 (NEW)
    vocab_size=VOCAB_SIZE
)

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
# Optimizer & Loss
# =============================================================================

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

print("Optimizer: Adam")
print(f"Learning rate: {LR}")
print("Loss: CrossEntropyLoss")
print()

# =============================================================================
# Training Loop with Early Stopping
# =============================================================================

print("="*70)
print("TRAINING START")
print("="*70)
print()

best_test_loss = float('inf')
best_test_acc = 0.0
patience_counter = 0

print(f"{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15} {'Test Acc':<15} {'Status':<20}")
print("-" * 75)

for epoch in range(EPOCHS):
    # =========================================================================
    # Training Phase
    # =========================================================================
    encoder.train()
    decoder.train()
    
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0
    
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

        train_loss += loss.item()
        train_batches += 1

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)          # [B, G]
            batch_acc = (preds == gap).float().mean().item()
            train_acc += batch_acc
    
    train_loss /= train_batches
    train_acc /= train_batches
    
    # =========================================================================
    # Evaluation Phase
    # =========================================================================
    encoder.eval()
    decoder.eval()
    
    test_loss = 0.0
    test_acc = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for left, right, gap in test_loader:
            left = left.to(device)
            right = right.to(device)
            gap = gap.to(device)

            # Concatenate flanks and transpose
            flanks = torch.cat([left, right], dim=1)
            flanks = flanks.permute(0, 2, 1)

            # Encode
            ctx = encoder(flanks)

            # Prepare decoder input
            start = torch.zeros(gap.size(0), 1,
                                dtype=torch.long, device=device)
            tgt_in = torch.cat([start, gap[:, :-1]], dim=1)

            # Decode
            logits = decoder(ctx, tgt_in)

            # Loss
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                gap.reshape(-1)
            )
            
            test_loss += loss.item()
            test_batches += 1

            # Accuracy
            preds = logits.argmax(dim=-1)
            batch_acc = (preds == gap).float().mean().item()
            test_acc += batch_acc
    
    test_loss /= test_batches
    test_acc /= test_batches
    
    # =========================================================================
    # Early Stopping & Checkpointing
    # =========================================================================
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_test_acc = test_acc
        patience_counter = 0
        
        # Save best models
        encoder_state = encoder.module.state_dict() if use_multi_gpu else encoder.state_dict()
        decoder_state = decoder.module.state_dict() if use_multi_gpu else decoder.state_dict()
        
        torch.save(encoder_state, 'encoder_best.pth')
        torch.save(decoder_state, 'decoder_best.pth')
        status = "✓ Best"
    else:
        patience_counter += 1
        status = f"patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {train_loss:<15.4f} {test_loss:<15.4f} {test_acc:<15.4f} {status:<20}")
    
    # Save regular checkpoint
    if (epoch + 1) % SAVE_EVERY == 0:
        encoder_state = encoder.module.state_dict() if use_multi_gpu else encoder.state_dict()
        decoder_state = decoder.module.state_dict() if use_multi_gpu else decoder.state_dict()
        
        torch.save(encoder_state, 'encoder.pth')
        torch.save(decoder_state, 'decoder.pth')
    
    # Early stopping check
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
        print(f"  Best test loss: {best_test_loss:.4f}")
        print(f"  Best test acc: {best_test_acc:.4f}")
        break

# =============================================================================
# Load Best Models & Save
# =============================================================================

encoder_module = encoder.module if use_multi_gpu else encoder
decoder_module = decoder.module if use_multi_gpu else decoder

encoder_module.load_state_dict(torch.load('encoder_best.pth'))
decoder_module.load_state_dict(torch.load('decoder_best.pth'))

torch.save(encoder_module.state_dict(), 'encoder.pth')
torch.save(decoder_module.state_dict(), 'decoder.pth')

# =============================================================================
# Training Complete
# =============================================================================

print()
print("="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Total epochs: {epoch+1}")
print(f"Best test loss: {best_test_loss:.4f}")
print(f"Best test accuracy: {best_test_acc:.4f}")
print()
print("Saved models:")
print("  encoder.pth, decoder.pth (best checkpoint)")
print("  encoder_best.pth, decoder_best.pth (backup)")
print("="*70)

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("\n✓ GPU memory cleared")
