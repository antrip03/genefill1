# build_samples.py - E. coli Only Training Data

import pickle
from utils.encoding import load_fasta

def make_samples(genome, flank_len=100, gap_len=50, n_samples=20000):
    """
    Extract random gap-filling samples from a single genome.
    
    Args:
        genome: Full genome sequence string
        flank_len: Length of flanking sequences (default 200bp)
        gap_len: Length of gap to predict (default 50bp)
        n_samples: Number of samples to generate
    
    Returns:
        List of (left_flank, right_flank, gap) tuples
    """
    import random
    samples = []
    max_start = len(genome) - (2*flank_len + gap_len)
    
    print(f"Genome length: {len(genome):,} bp")
    print(f"Max valid start position: {max_start:,}")
    print(f"Generating {n_samples:,} samples...")
    
    for i in range(n_samples):
        s = random.randint(0, max_start)
        
        left = genome[s : s + flank_len]
        gap = genome[s + flank_len : s + flank_len + gap_len]
        right = genome[s + flank_len + gap_len : s + 2*flank_len + gap_len]
        
        # Skip samples with ambiguous bases (N, n, etc.)
        if 'N' in left.upper() or 'N' in right.upper() or 'N' in gap.upper():
            continue
        
        samples.append((left, right, gap))
        
        if (i + 1) % 2000 == 0:
            print(f"  Generated {len(samples):,} valid samples...")
    
    print(f"✓ Total valid samples: {len(samples):,}")
    return samples


if __name__ == "__main__":
    print("="*70)
    print("E. COLI TRAINING DATA GENERATION")
    print("="*70)
    print()
    
    # Load E. coli genome
    print("Loading E. coli genome...")
    genome = load_fasta("data/raw/GCF_000005845.2_ASM584v2_genomic.fna")
    
    # Calculate GC content
    gc_count = genome.upper().count('G') + genome.upper().count('C')
    gc_percent = (gc_count / len(genome)) * 100
    print(f"GC content: {gc_percent:.2f}%")
    print()
    
    # Generate samples
    samples = make_samples(genome, flank_len=100, gap_len=50, n_samples=20000)
    
    # Save to pickle
    output_path = "data/processed/ecoli_gapfill_samples.pkl"
    print(f"\nSaving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    
    print("✓ Done!")
    print()
    print("="*70)
    print("NEXT STEPS:")
    print("  1. Update train.py to load 'ecoli_gapfill_samples.pkl'")
    print("  2. Run: python train.py")
    print("="*70)

