import torch

NUCLEOTIDES = ["A", "C", "G", "T"]

# NEW: index mapping and special tokens for masked LM
BASE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}
PAD_IDX = 4      # padding token
MASK_IDX = 5     # masked token
VOCAB_SIZE = 6   # A,C,G,T, PAD, MASK

def dna_to_one_hot(seq: str):
    one_hot = []
    for base in seq.upper():
        row = [0, 0, 0, 0]
        if base in NUCLEOTIDES:
            row[NUCLEOTIDES.index(base)] = 1
        one_hot.append(row)
    return torch.tensor(one_hot, dtype=torch.float32)

def one_hot_to_dna(arr):
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    seq = []
    for row in arr:
        idx = row.index(1)
        seq.append(NUCLEOTIDES[idx])
    return "".join(seq)

# NEW: DNA → integer indices (0–3 for A,C,G,T; MASK for anything else)
def dna_to_indices(seq: str):
    idxs = []
    for b in seq.upper():
        if b in BASE_TO_INDEX:
            idxs.append(BASE_TO_INDEX[b])
        else:
            idxs.append(MASK_IDX)
    return torch.tensor(idxs, dtype=torch.long)

def load_fasta(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seqs.append(line.strip().upper())
    return "".join(seqs)
