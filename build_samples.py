import pickle
from utils.encoding import load_fasta

def make_samples(genome, flank_len=200, gap_len=50, n_samples=50000):
    import random
    samples = []
    max_start = len(genome) - (2*flank_len + gap_len)
    for _ in range(n_samples):
        s = random.randint(0, max_start)
        left = genome[s : s + flank_len]
        gap = genome[s + flank_len : s + flank_len + gap_len]
        right = genome[s + flank_len + gap_len : s + 2*flank_len + gap_len]
        samples.append((left, right, gap))
    return samples

if __name__ == "__main__":
    genome = load_fasta("data/raw/GCA_000027325.1_ASM2732v1_genomic.fna")
    samples = make_samples(genome)
    with open("data/processed/gapfill_samples.pkl", "wb") as f:
        pickle.dump(samples, f)
