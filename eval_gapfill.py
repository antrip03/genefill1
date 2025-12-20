# eval_gapfill.py - Evaluation with Wave Beam Search (BiLSTM Decoder + 512-dim)

import torch
import torch.nn.functional as F
import pickle
from collections import OrderedDict

from models import CNNBiLSTMEncoder, GapDecoder
from utils.dataset import GapFillDataset
from utils.encoding import NUCLEOTIDES

# =============================================================================
# Hyperparameters (MUST MATCH train.py)
# =============================================================================

FLANK_LEN = 200
GAP_LEN = 50
CONTEXT_DIM = 512        # 512-dim
HIDDEN_SIZE = 512        # 512-dim
LSTM_HIDDEN = 256        # BiLSTM 256
VOCAB_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Wave Beam Search
# =============================================================================

BASES = NUCLEOTIDES

def wave_beam_search(decoder, ctx, gap_len, beam_size=5, beam_expand=10, wave_period=5):
    """
    Wave-beam search: expand beam periodically to escape local optima.
    
    Args:
        decoder: GapDecoder module
        ctx: [batch, context_dim] encoder context
        gap_len: Length of gap to generate
        beam_size: Normal beam width
        beam_expand: Expanded beam width (on wave period)
        wave_period: How often to expand
    """
    batch_size = ctx.size(0)
    candidates = [([], 0.0) for _ in range(batch_size)]  # (seq_indices, log_prob)

    for step in range(gap_len):
        # Decide beam width for this step
        cur_beam = beam_expand if (step > 0 and step % wave_period == 0) else beam_size
        new_cands = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            for seq, logp in candidates[b]:
                # Build input: [START_TOKEN] + previous_tokens
                prev = torch.tensor([[0] + seq], device=device, dtype=torch.long)
                
                with torch.no_grad():
                    logits = decoder(ctx[b:b+1], prev)[:, -1, :]  # [1, vocab]
                    logp_step = F.log_softmax(logits[0], dim=-1)  # [vocab]

                # Expand to all vocabulary tokens
                for token_idx in range(VOCAB_SIZE):
                    new_seq = seq + [token_idx]
                    new_logp = logp + logp_step[token_idx].item()
                    new_cands[b].append((new_seq, new_logp))

            # Keep top-k candidates
            new_cands[b].sort(key=lambda x: x[1], reverse=True)
            candidates[b] = new_cands[b][:cur_beam]

    # Extract best sequence for each batch element
    best_seqs = []
    for b in range(batch_size):
        best_seq, _ = candidates[b][0]
        best_seqs.append(best_seq)
    
    # Convert to tensor [batch, gap_len]
    result = torch.tensor(best_seqs, device=device, dtype=torch.long)
    return result


def greedy_decode(decoder, ctx, gap_len):
    """
    Greedy decoding: at each step, choose the token with highest probability.
    
    Args:
        decoder: GapDecoder module
        ctx: [batch, context_dim] encoder context
        gap_len: Length of gap to generate
    """
    batch_size = ctx.size(0)
    sequences = torch.zeros((batch_size, gap_len), device=device, dtype=torch.long)
    
    for step in range(gap_len):
        # Build input: [START_TOKEN] + previous_tokens
        prev = torch.cat([
            torch.zeros((batch_size, 1), device=device, dtype=torch.long),
            sequences[:, :step]
        ], dim=1)
        
        with torch.no_grad():
            logits = decoder(ctx, prev)[:, -1, :]  # [batch, vocab]
            tokens = logits.argmax(dim=-1)  # [batch]
        
        sequences[:, step] = tokens
    
    return sequences


# =============================================================================
# Load Dataset
# =============================================================================

print("Loading dataset...")
with open("data/processed/ecoli_gapfill_samples.pkl", "rb") as f:
    all_samples = pickle.load(f)

test_samples = all_samples[-1000:]  # Same test split as training
test_dataset = GapFillDataset(test_samples)
print(f"Test samples: {len(test_samples)}\n")

# =============================================================================
# Load Models (Corrected for BiLSTM Decoder + 512-dim)
# =============================================================================

print("Initializing models...")

encoder = CNNBiLSTMEncoder(
    in_channels=4,
    hidden_channels=128,      # 3-layer CNN
    lstm_hidden=LSTM_HIDDEN,  # 256
    context_dim=CONTEXT_DIM   # 512
).to(device)

decoder = GapDecoder(
    context_dim=CONTEXT_DIM,  # 512
    hidden_size=HIDDEN_SIZE,  # 512
    vocab_size=VOCAB_SIZE
).to(device)

print("Loading weights...")

# Handle DataParallel keys if present
def fix_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict

enc_state = torch.load("encoder.pth", map_location=device)
dec_state = torch.load("decoder.pth", map_location=device)

encoder.load_state_dict(fix_keys(enc_state))
decoder.load_state_dict(fix_keys(dec_state))

encoder.eval()
decoder.eval()
print("✓ Models loaded\n")

# =============================================================================
# Helper Functions
# =============================================================================

def idx_to_seq(idx_tensor):
    """Convert index tensor to nucleotide sequence."""
    if idx_tensor.dim() > 1:
        idx_tensor = idx_tensor.squeeze()
    return "".join(NUCLEOTIDES[i] for i in idx_tensor.cpu().tolist())

def gap_accuracy(true_idx, pred_idx):
    """Compute token-wise and exact-match accuracy."""
    true_flat = true_idx.flatten()
    pred_flat = pred_idx.flatten()
    
    # Token-wise accuracy
    min_len = min(true_flat.size(0), pred_flat.size(0))
    token_acc = (true_flat[:min_len] == pred_flat[:min_len]).float().mean().item()
    
    # Exact match
    exact = float(
        true_flat.size(0) == pred_flat.size(0) and 
        (true_flat == pred_flat).all().item()
    )
    
    return token_acc, exact

# =============================================================================
# Evaluate
# =============================================================================

USE_BEAM_SEARCH = True  # Set to False for greedy decoding

total_token_acc = 0.0
total_exact = 0.0
n_samples = len(test_dataset)

method = "WAVE BEAM SEARCH" if USE_BEAM_SEARCH else "GREEDY"

print("="*70)
print(f"EVALUATION: {n_samples} TEST SAMPLES")
print(f"Method: {method}")
print("="*70)
print()

for idx in range(n_samples):
    left, right, gap_true = test_dataset[idx]
    
    # Prepare input: [1, 4, 400]
    left = left.permute(1, 0).unsqueeze(0).to(device)
    right = right.permute(1, 0).unsqueeze(0).to(device)
    gap_true = gap_true.unsqueeze(0).to(device)
    
    flanks = torch.cat([left, right], dim=2)
    
    # Encode
    with torch.no_grad():
        ctx = encoder(flanks)
        
        # Decode
        if USE_BEAM_SEARCH:
            gap_pred = wave_beam_search(decoder, ctx, GAP_LEN, beam_size=5)
        else:
            gap_pred = greedy_decode(decoder, ctx, GAP_LEN)
    
    # Evaluate
    token_acc, exact = gap_accuracy(gap_true, gap_pred)
    total_token_acc += token_acc
    total_exact += exact
    
    # Print first 10
    if idx < 10:
        print(f"Example {idx+1}:")
        print(f"  TRUE: {idx_to_seq(gap_true)}")
        print(f"  PRED: {idx_to_seq(gap_pred)}")
        print(f"  Token Acc: {token_acc:.4f}")
        print()

print("="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Decoding: {method}")
print(f"Test samples: {n_samples}")
print(f"Avg Token Accuracy: {total_token_acc / n_samples:.4f}")
print(f"Exact Match Rate: {total_exact / n_samples:.6f}")
print("="*70)
