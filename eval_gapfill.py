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
    # each candidate: (list_of_indices, cumulative_log_prob)
    candidates = [([], 0.0)]

    for step in range(gap_len):
        cur_beam = beam_expand if (step > 0 and step % wave_period == 0) else beam_size
        new_cands = []

        for seq, logp in candidates:
            # previous token sequence (start_token=0 prepended)
            prev = torch.tensor([[0] + seq], device=ctx.device)  # shape (1, len+1)

            with torch.no_grad():
                # assume decoder(ctx, prev) -> logits (1, T, vocab)
                logits = decoder(ctx, prev)[:, -1, :]             # last step, (1, vocab)
                logp_step = F.log_softmax(logits[0], dim=-1)      # (vocab,)

            for i in range(len(BASES)):
                new_seq = seq + [i]
                new_logp = logp + logp_step[i].item()
                new_cands.append((new_seq, new_logp))

        new_cands.sort(key=lambda x: x[1], reverse=True)
        candidates = new_cands[:cur_beam]

    best_seq, _ = candidates[0]
    return torch.tensor(best_seq, device=ctx.device).unsqueeze(0)  # (1, gap_len)


# ---------- Load data ----------

with open("data/processed/gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

dataset = GapFillDataset(samples)

left, right, gap_idx = dataset[0]
left = left.unsqueeze(0).to(device)      # (1, 4, FLANK_LEN)
right = right.unsqueeze(0).to(device)    # (1, 4, FLANK_LEN)
gap_idx = gap_idx.unsqueeze(0).to(device)

# ---------- Load model ----------

encoder = CNNEncoder(4, 128, CONTEXT_DIM).to(device)
decoder = GapDecoder(CONTEXT_DIM, HIDDEN_SIZE, VOCAB_SIZE).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
encoder.eval()
decoder.eval()

# ---------- Build context and decode ----------

flanks = torch.cat([left, right], dim=1)   # (1, 4, 2*FLANK_LEN)
flanks = flanks.permute(0, 2, 1)          # (1, 2*FLANK_LEN, 4) if encoder expects that

with torch.no_grad():
    ctx = encoder(flanks)                  # context for decoder
    # greedy (old): pred_idx = decoder.generate(ctx, max_len=GAP_LEN, start_token=0)
    pred_idx = wave_beam_search(
        decoder,
        ctx,
        GAP_LEN,
        beam_size=5,
        beam_expand=10,
        wave_period=5,
    )

# ---------- Helpers & print ----------

def idx_to_seq(idx_tensor):
    idx_list = idx_tensor.squeeze(0).tolist()
    return "".join(NUCLEOTIDES[i] for i in idx_list)

true_gap = idx_to_seq(gap_idx)
pred_gap = idx_to_seq(pred_idx)

print("TRUE GAP : ", true_gap)
print("PRED GAP : ", pred_gap)
