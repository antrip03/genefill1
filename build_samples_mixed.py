"""
Build mixed training dataset from multiple genomes
Combines E. coli and Streptomyces for balanced G/C training
"""

import random
import pickle
from collections import Counter

def load_genome(fasta_path):
    """Load genome from FASTA file"""
    print(f"Loading {fasta_path}...")
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    
    # Skip headers, join sequence
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    sequence = sequence.upper()
    
    print(f"  Loaded {len(sequence):,} bp")
    return sequence

def compute_gc_content(sequence):
    """Calculate G/C percentage"""
    gc = sequence.count('G') + sequence.count('C')
    return gc / len(sequence)

def create_samples(genome, n_samples, gap_length=50, flank_length=100, genome_name=""):
    """
    Create training samples from a genome
    Returns: list of (left_flank, gap, right_flank)
    """
    print(f"\nCreating {n_samples} samples from {genome_name}...")
    
    samples = []
    max_start = len(genome) - (2 * flank_length + gap_length)
    
    if max_start <= 0:
        print(f"  Error: Genome too short!")
        return []
    
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        
        if attempts % 5000 == 0:
            print(f"  Progress: {len(samples)}/{n_samples} samples")
        
        # Random position
        start = random.randint(0, max_start)
        
        left_start = start
        left_end = start + flank_length
        gap_start = left_end
        gap_end = gap_start + gap_length
        right_start = gap_end
        right_end = right_start + flank_length
        
        left_flank = genome[left_start:left_end]
        gap = genome[gap_start:gap_end]
        right_flank = genome[right_start:right_end]
        
        # Quality check: only ACGT, no N's
        window = left_flank + gap + right_flank
        if all(base in 'ACGT' for base in window):
            samples.append((left_flank, gap, right_flank))
    
    print(f"  Created {len(samples)} valid samples")
    return samples

def analyze_samples(samples, name="Dataset"):
    """Print statistics about the samples"""
    all_gaps = ''.join(gap for _, gap, _ in samples)
    counter = Counter(all_gaps)
    total = len(all_gaps)
    
    print(f"\n{name} Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total bases in gaps: {total:,}")
    print(f"\n  Base distribution:")
    for base in ['A', 'C', 'G', 'T']:
        count = counter.get(base, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"    {base}: {pct:5.1f}% ({count:,})")
    
    gc_pct = (counter['G'] + counter['C']) / total * 100 if total > 0 else 0
    print(f"\n  G/C content: {gc_pct:.1f}%")

def main():
    print("="*70)
    print("BUILDING MIXED TRAINING DATASET")
    print("="*70)
    
    # Configuration
    genomes = [
        {
            'path': 'data/raw/GCA_000027325.1_ASM2732v1_genomic.fna',
            'name': 'E. coli',
            'n_samples': 50000
        },
        {
            'path': 'data/raw/GCF_000203835.1_ASM20383v1_genomic.fna',
            'name': 'Streptomyces',
            'n_samples': 50000
        }
    ]
    
    gap_length = 50
    flank_length = 100
    
    # Load genomes and create samples
    all_samples = []
    
    for genome_config in genomes:
        print(f"\n{'='*70}")
        print(f"Processing: {genome_config['name']}")
        print('='*70)
        
        # Load genome
        genome = load_genome(genome_config['path'])
        
        # Calculate G/C content
        gc = compute_gc_content(genome)
        print(f"  G/C content: {gc:.1%}")
        
        # Create samples
        samples = create_samples(
            genome,
            n_samples=genome_config['n_samples'],
            gap_length=gap_length,
            flank_length=flank_length,
            genome_name=genome_config['name']
        )
        
        # Analyze
        analyze_samples(samples, genome_config['name'])
        
        # Add to master list
        all_samples.extend(samples)
    
    # Shuffle mixed samples
    print(f"\n{'='*70}")
    print("COMBINING DATASETS")
    print('='*70)
    print(f"Total samples before shuffle: {len(all_samples)}")
    
    random.shuffle(all_samples)
    
    print(f"Total samples after shuffle: {len(all_samples)}")
    
    # Analyze mixed dataset
    analyze_samples(all_samples, "Mixed Dataset")
    
    # Save
    output_file = "data/processed/mixed_gapfill_samples.pkl"
    print(f"\nSaving to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_samples, f)
    
    print("✓ Done!")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Training samples: {len(all_samples):,}")
    print(f"Gap length: {gap_length} bp")
    print(f"Flank length: {flank_length} bp each side")
    print(f"Genomes: {', '.join(g['name'] for g in genomes)}")
    print(f"Output: {output_file}")
    print('='*70)

if __name__ == "__main__":
    main()
