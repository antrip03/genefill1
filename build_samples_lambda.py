# build_samples_lambda.py
import urllib.request
import pickle
import os

# Stable NCBI URL for Enterobacteria phage lambda (NC_001416.1)
URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001416.1&rettype=fasta&retmode=text"
FILENAME = "data/raw/lambda_phage.fasta"

def load_fasta(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"): continue
            seqs.append(line.strip().upper())
    return "".join(seqs)

def make_real_samples(genome, flank_len=50, gap_len=10, n_samples=5000):
    import random
    samples = []
    max_start = len(genome) - (2*flank_len + gap_len)
    for _ in range(n_samples):
        s = random.randint(0, max_start)
        left = genome[s : s + flank_len]
        gap = genome[s + flank_len : s + flank_len + gap_len]
        right = genome[s + flank_len + gap_len : s + flank_len + gap_len + flank_len]
        samples.append((left, right, gap))
    return samples

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print(f"Downloading Lambda Phage genome from NCBI...")
    urllib.request.urlretrieve(URL, FILENAME)
    print("Download complete.")
    
    genome = load_fasta(FILENAME)
    print(f"Genome length: {len(genome)} bp")
    
    # Generate simple samples (50bp flank, 10bp gap)
    samples = make_real_samples(genome, flank_len=50, gap_len=10, n_samples=5000)
    
    out_path = "data/processed/lambda_gapfill_samples.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    
    print(f"✓ Saved 5000 real Lambda Phage samples to {out_path}")
