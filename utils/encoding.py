import torch

NUCLEOTIDES = ["A", "C", "G", "T"]

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

def load_fasta(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seqs.append(line.strip().upper())
    return "".join(seqs)
