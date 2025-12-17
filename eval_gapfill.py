import torch
import torch.nn.functional as F
import pickle

from models import CNNEncoder, GapDecoder
from utils.dataset import GapFillDataset
from utils.encoding import NUCLEOTIDES

FLANK_LEN = 200
GAP_LEN = 50
CONTEXT_DIM = 256
HIDDEN_SIZE = 256
VOCAB_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Greedy decoding ----------

def greedy_decode(decoder, ctx, gap_len):
    """Simple greedy decoding without beam search."""
    with torch.no_grad():
        pred_idx = decoder.generate(ctx, gap_len, start_token=0)
    return pred_idx


# ---------- Load dataset ----------

with open("data/processed/gapfill_samples.pkl", "rb") as f:
    all_samples = pickle.load(f)

# Split into train and test
TEST_SIZE = 1000 if len(all_samples) > 1000 else len(all_samples)
test_samples = all_samples[-TEST_SIZE:]
test_dataset = GapFillDataset(test_samples)


# ---------- Load model ----------

encoder = CNNEncoder(4, 128, CONTEXT_DIM).to(device)
decoder = GapDecoder(CONTEXT_DIM, HIDDEN_SIZE, VOCAB_SIZE).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
encoder.eval()
decoder.eval()


# ---------- Helpers ----------

def idx_to_seq(idx_tensor):
    idx_list = idx_tensor.squeeze(0).tolist()
    return "".join(NUCLEOTIDES[i] for i in idx_list)

def gap_accuracy(true_idx, pred_idx):
    t = true_idx.squeeze(0)
    p = pred_idx.squeeze(0)
    L = min(t.size(0), p.size(0))
    matches = (t[:L] == p[:L]).float().mean().item()
    exact = float(L == t.size(0) == p.size(0) and matches == 1.0)
    return matches, exact


# ---------- Evaluate over many gaps ----------

total_token_acc = 0.0
total_exact = 0.0
n = len(test_dataset)

print(f"Evaluating on {n} test samples using GREEDY decoding...\n")

for i in range(n):
    left, right, gap_idx = test_dataset[i]

    # left and right are [FLANK_LEN, 4] from dataset
    # Need to add batch dim and transpose to [1, 4, FLANK_LEN]
    left = left.permute(1, 0).unsqueeze(0).to(device)      # [1, 4, FLANK_LEN]
    right = right.permute(1, 0).unsqueeze(0).to(device)    # [1, 4, FLANK_LEN]
    gap_idx = gap_idx.unsqueeze(0).to(device)

    # Concatenate along sequence dimension
    flanks = torch.cat([left, right], dim=2)   # [1, 4, 2*FLANK_LEN]

    with torch.no_grad():
        ctx = encoder(flanks)
        pred_idx = greedy_decode(decoder, ctx, GAP_LEN)

    token_acc, exact = gap_accuracy(gap_idx, pred_idx)
    total_token_acc += token_acc
    total_exact += exact

    if i < 5:  # print first 5 examples
        print(f"Example {i}")
        print("TRUE:", idx_to_seq(gap_idx))
        print("PRED:", idx_to_seq(pred_idx))
        print("token_acc:", f"{token_acc:.4f}")
        print()

print("=" * 60)
print(f"Num test gaps: {n}")
print(f"Avg token-wise accuracy: {total_token_acc / n:.4f}")
print(f"Exact-gap match rate: {total_exact / n:.6f}")
print("=" * 60)
