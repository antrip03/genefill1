import urllib.request
import pickle
import os
import random

# E. coli K-12 MG1655 chromosome
URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text"
)
FILENAME = "data/raw/ecoli_k12.fasta"

FLANK_LEN = 100   # easier, shorter context
GAP_LEN = 10
N_SAMPLES = 20000
OUT_PATH = "data/processed/ecoli_gapfill_easy_20k_100_10.pkl"


def load_fasta(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seqs.append(line.strip().upper())
    return "".join(seqs)


def make_real_samples(genome, flank_len=FLANK_LEN, gap_len=GAP_LEN, n_samples=N_SAMPLES):
    samples = []
    max_start = len(genome) - (2 * flank_len + gap_len)
    for _ in range(n_samples):
        s = random.randint(0, max_start)
        left = genome[s : s + flank_len]
        gap = genome[s + flank_len : s + flank_len + gap_len]
        right = genome[s + flank_len + gap_len : s + flank_len + gap_len + flank_len]
        samples.append((left, right, gap))
    return samples


if __name__ == "__main__":
    random.seed(42)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(FILENAME):
        print("Downloading E. coli K-12 genome from NCBI...")
        urllib.request.urlretrieve(URL, FILENAME)
        print("Download complete.")
    else:
        print("E. coli FASTA already exists, skipping download.")

    genome = load_fasta(FILENAME)
    print("Genome length:", len(genome))

    print(f"Generating {N_SAMPLES} samples ({FLANK_LEN}bp flanks, {GAP_LEN}bp gap)...")
    samples = make_real_samples(genome)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(samples, f)
    print(f"✓ Saved samples to {OUT_PATH}")
