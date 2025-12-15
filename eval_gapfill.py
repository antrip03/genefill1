import torch
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

with open("data/processed/gapfill_samples.pkl", "rb") as f:
    samples = pickle.load(f)

dataset = GapFillDataset(samples)

left, right, gap_idx = dataset[0]
left = left.unsqueeze(0).to(device)
right = right.unsqueeze(0).to(device)
gap_idx = gap_idx.unsqueeze(0).to(device)

encoder = CNNEncoder(4, 128, CONTEXT_DIM).to(device)
decoder = GapDecoder(CONTEXT_DIM, HIDDEN_SIZE, VOCAB_SIZE).to(device)
encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
encoder.eval()
decoder.eval()

flanks = torch.cat([left, right], dim=1)
flanks = flanks.permute(0, 2, 1)

with torch.no_grad():
    ctx = encoder(flanks)
    pred_idx = decoder.generate(ctx, max_len=GAP_LEN, start_token=0)

def idx_to_seq(idx_tensor):
    idx_list = idx_tensor.squeeze(0).tolist()
    return "".join(NUCLEOTIDES[i] for i in idx_list)

true_gap = idx_to_seq(gap_idx)
pred_gap = idx_to_seq(pred_idx)

print("TRUE GAP : ", true_gap)
print("PRED GAP : ", pred_gap)
