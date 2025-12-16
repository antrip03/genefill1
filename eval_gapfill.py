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

# ---------- Wave-beam search ----------

BASES = NUCLEOTIDES
idx_to_base = {i: b for i, b in enumerate(BASES)}

def wave_beam_search(decoder, ctx, gap_len,
                     beam_size=5, beam_expand=10, wave_period=5):
    """
    Wave-beam search over decoder outputs.
    decoder(ctx, prev_tokens) -> logits over vocab at each step.
    """
    candidates = [([], 0.0)]  # (sequence_indices, cumulative_log_prob)

    for step in range(gap_len):
        cur_beam = beam_expand if (step > 0 and step % wave_period == 0) else beam_size
        new_cands = []

        for seq, logp in candidates:
            prev = torch.tensor([[0] + seq], device=ctx.device)  # (1, len+1)

            with torch.no_grad():
                logits = decoder(ctx, prev)[:, -1, :]          # (1, vocab)
                logp_step = F.log_softmax(logits[0], dim=-1)   # (vocab,)

            for i in range(len(BASES)):
                new_seq = seq + [i]
                new_logp = logp + logp_step[i].item()
                new_cands.append((new_seq, new_logp))

        new_cands.sort(key=lambda x: x[1], reverse=True)
        candidates = new_cands[:cur_beam]

    best_seq, _ = candidates[0]
    return torch.tensor(best_seq, device=ctx.device).unsqueeze(0)  # (1, gap_len)


# ---------- Load dataset ----------

with open("data/processed/gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

dataset = GapFillDataset(samples)

TEST_SIZE = 1000 if len(dataset) > 1000 else len(dataset)
test_dataset = dataset[-TEST_SIZE:]  # simple hold-out slice


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

for i in range(n):
    left, right, gap_idx = test_dataset[i]

    left = left.unsqueeze(0).to(device)      # (1, 4, FLANK_LEN)
    right = right.unsqueeze(0).to(device)    # (1, 4, FLANK_LEN)
    gap_idx = gap_idx.unsqueeze(0).to(device)

    flanks = torch.cat([left, right], dim=1)   # (1, 4, 2*FLANK_LEN)
    flanks = flanks.permute(0, 2, 1)          # (1, 2*FLANK_LEN, 4)

    with torch.no_grad():
        ctx = encoder(flanks)
        pred_idx = wave_beam_search(
            decoder, ctx, GAP_LEN,
            beam_size=5, beam_expand=10, wave_period=5,
        )

    token_acc, exact = gap_accuracy(gap_idx, pred_idx)
    total_token_acc += token_acc
    total_exact += exact

    if i < 3:  # print a few qualitative examples
        print(f"Example {i}")
        print("TRUE:", idx_to_seq(gap_idx))
        print("PRED:", idx_to_seq(pred_idx))
        print("token_acc:", token_acc)
        print()

print("Num test gaps:", n)
print("Avg token-wise accuracy:", total_token_acc / n)
print("Exact-gap match rate:", total_exact / n)
