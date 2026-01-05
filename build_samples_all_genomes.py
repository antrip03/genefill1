# build_samples_all_genomes.py
# Generate real flank–gap–flank samples for multiple bacterial genomes
# EXACT SAME PATTERN as the provided E. coli K‑12 script, replicated per genome.

import urllib.request
import pickle
import os
import random

# ------------------------------------------------------------------
# URLs and output paths for all 7 genomes
# ------------------------------------------------------------------

GENOMES = [
    # (label, fasta_url, fasta_filename, out_pickle_path)
    (
        "ECOR",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_007779.1&rettype=fasta&retmode=text",  # example ECOR ref [web:82]
        "data/raw/ecor.fasta",
        "data/processed/ecor_gapfill_samples.pkl",
    ),
    (
        "E. coli K-12 MG1655",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text",  # MG1655 [web:53]
        "data/raw/ecoli_k12.fasta",
        "data/processed/MG1655_gapfill_samples.pkl",
    ),
    (
        "UTI89",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_007946.1&rettype=fasta&retmode=text",  # UTI89 chromosome [web:82]
        "data/raw/uti89.fasta",
        "data/processed/uti89_gapfill_samples.pkl",
    ),
    (
        "Shigella flexneri",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_004741.1&rettype=fasta&retmode=text",  # S. flexneri 2a 301 [web:82]
        "data/raw/shigella.fasta",
        "data/processed/shigella_gapfill_samples.pkl",
    ),
    (
        "Salmonella enterica",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_003197.2&rettype=fasta&retmode=text",  # S. Typhimurium LT2 [web:53]
        "data/raw/salmonella.fasta",
        "data/processed/salmonella_gapfill_samples.pkl",
    ),
    (
        "Enterobacter cloacae",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_014121.1&rettype=fasta&retmode=text",  # E. cloacae subsp. cloacae [web:82]
        "data/raw/clocae.fasta",
        "data/processed/clocae_gapfill_samples.pkl",
    ),
    (
        "Klebsiella pneumoniae",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=nuccore&id=NC_016845.1&rettype=fasta&retmode=text",  # K. pneumoniae HS11286 [web:82]
        "data/raw/klebsiella.fasta",
        "data/processed/klebsiella_samples.pkl",
    ),
]

# EXACT SAME GLOBAL SETTINGS AS YOUR SCRIPT
FLANK_LEN = 100   # easier, shorter context
GAP_LEN = 10
N_SAMPLES = 20000


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

    for label, url, fasta_path, out_path in GENOMES:
        print("=" * 70)
        print(f"PROCESSING GENOME: {label}")
        print("=" * 70)

        if not os.path.exists(fasta_path):
            print(f"Downloading {label} genome from NCBI...")
            urllib.request.urlretrieve(url, fasta_path)
            print("Download complete.")
        else:
            print(f"{label} FASTA already exists, skipping download.")

        genome = load_fasta(fasta_path)
        print("Genome length:", len(genome))

        print(f"Generating {N_SAMPLES} samples ({FLANK_LEN}bp flanks, {GAP_LEN}bp gap)...")
        samples = make_real_samples(genome)
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"✓ Saved samples to {out_path}")
        print()
