# train.py - E. COLI ONLY TRAINING (3-LAYER CNN + SCALED 512-DIM)
# Proven working architecture with increased capacity

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CNNBiLSTMEncoder, GapDecoder
from utils.dataset import GapFillDataset
import time

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
# Hyperparameters (3-LAYER CNN + SCALED 512-DIM)
# =============================================================================

BATCH_SIZE = 32           # Proven working
LR = 5e-3                 # Proven working
EPOCHS = 100              # More epochs for scaling
FLANK_LEN = 300
GAP_LEN = 50
CONTEXT_DIM = 1024         # SCALED UP
HIDDEN_SIZE = 1024         # SCALED UP
VOCAB_SIZE = 4
LSTM_HIDDEN = 256         # SCALED UP (was 128)
EARLY_STOPPING_PATIENCE = 30

# Multi-GPU settings
USE_MULTI_GPU = True
SAVE_EVERY = 10

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
        BATCH_SIZE = BATCH_SIZE * num_gpus
        print(f"  Scaled batch size: {BATCH_SIZE}")
    else:
        print(f"✓ Single GPU mode: {torch.cuda.get_device_name(0)}")

print()
print("="*70)
print("TRAINING CONFIGURATION (E. COLI, 3-LAYER CNN + 512-DIM SCALED)")
print("="*70)
print(f"  Dataset: E. coli K-12 (50.8% GC, Balanced)")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"  Context Dim: {CONTEXT_DIM}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  LSTM Hidden: {LSTM_HIDDEN}")
print(f"  Architecture: CNN(3-layer, 128ch) + BiLSTM({LSTM_HIDDEN}dim) Encoder")
print(f"                + 2-Layer LSTM({HIDDEN_SIZE}dim) Decoder")
print(f"  Multi-GPU: {'Yes' if use_multi_gpu else 'No'}")
print(f"  Optimizer: Adam (NO weight decay)")
print("="*70)
print()

# =============================================================================
# Data Loading (E. coli Only)
# =============================================================================

print("Loading E. coli dataset...")
with open("data/processed/ecoli_gapfill_samples.pkl", "rb") as f:
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
# Models (3-LAYER CNN + 512-DIM)
# =============================================================================

print("Initializing models (3-layer CNN + 512-dim)...")

encoder = CNNBiLSTMEncoder(
    in_channels=4,
    hidden_channels=128,      # 3-layer CNN output
    lstm_hidden=LSTM_HIDDEN,  # 256
    context_dim=CONTEXT_DIM   # 512
)

decoder = GapDecoder(
    context_dim=CONTEXT_DIM,  # 512
    hidden_size=HIDDEN_SIZE,  # 512
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
    lr=LR
)

criterion = nn.CrossEntropyLoss()

print("Optimizer: Adam (no weight decay)")
print(f"Learning rate: {LR}")
print("Loss: CrossEntropyLoss")
print()

# =============================================================================
# Training Loop with Batch Progress & Early Stopping
# =============================================================================

print("="*70)
print("TRAINING START")
print("="*70)
print()

best_test_loss = float('inf')
best_test_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    # =========================================================================
    # Training Phase
    # =========================================================================
    encoder.train()
    decoder.train()
    
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("-" * 70)
    
    for batch_idx, (left, right, gap) in enumerate(train_loader):
        batch_start_time = time.time()
        
        left = left.to(device)
        right = right.to(device)
        gap = gap.to(device)

        # Concatenate flanks and transpose to [B, 4, 2L]
        flanks = torch.cat([left, right], dim=1)
        flanks = flanks.permute(0, 2, 1)

        # Encode flanks with CNN + BiLSTM
        ctx = encoder(flanks)

        # Prepare decoder input with start token
        start = torch.zeros(gap.size(0), 1, dtype=torch.long, device=device)
        tgt_in = torch.cat([start, gap[:, :-1]], dim=1)

        # Decode to predict gap
        logits = decoder(ctx, tgt_in)

        # Compute loss
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), gap.reshape(-1))

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
            preds = logits.argmax(dim=-1)
            batch_acc = (preds == gap).float().mean().item()
            train_acc += batch_acc
        
        batch_time = time.time() - batch_start_time
        
        # Print batch progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = train_loss / train_batches
            avg_acc = train_acc / train_batches
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
                  f"Time: {batch_time:.2f}s")
    
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
    
    print()
    print("  Evaluating on test set...")
    
    with torch.no_grad():
        for batch_idx, (left, right, gap) in enumerate(test_loader):
            left = left.to(device)
            right = right.to(device)
            gap = gap.to(device)

            flanks = torch.cat([left, right], dim=1)
            flanks = flanks.permute(0, 2, 1)

            ctx = encoder(flanks)

            start = torch.zeros(gap.size(0), 1, dtype=torch.long, device=device)
            tgt_in = torch.cat([start, gap[:, :-1]], dim=1)

            logits = decoder(ctx, tgt_in)

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), gap.reshape(-1))
            
            test_loss += loss.item()
            test_batches += 1

            preds = logits.argmax(dim=-1)
            batch_acc = (preds == gap).float().mean().item()
            test_acc += batch_acc
    
    test_loss /= test_batches
    test_acc /= test_batches
    
    epoch_time = time.time() - epoch_start_time
    
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
        status = "✓ BEST MODEL SAVED"
    else:
        patience_counter += 1
        status = f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
    
    # Print epoch summary
    print()
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
    print(f"  Epoch Time: {epoch_time:.2f}s | {status}")
    print("="*70)
    print()
    
    # Save regular checkpoint
    if (epoch + 1) % SAVE_EVERY == 0:
        encoder_state = encoder.module.state_dict() if use_multi_gpu else encoder.state_dict()
        decoder_state = decoder.module.state_dict() if use_multi_gpu else decoder.state_dict()
        
        torch.save(encoder_state, 'encoder.pth')
        torch.save(decoder_state, 'decoder.pth')
        print(f"✓ Checkpoint saved at epoch {epoch+1}")
        print()
    
    # Early stopping check
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"✓ Early stopping triggered at epoch {epoch+1}")
        print(f"  Best test loss: {best_test_loss:.4f}")
        print(f"  Best test acc: {best_test_acc:.4f}")
        break

# =============================================================================
# Load Best Models & Save Final
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
