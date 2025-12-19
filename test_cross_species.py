"""
Zero-shot cross-species evaluation
Tests E. coli-trained model on G/C-rich Streptomyces genome
"""

import torch
import random
from collections import Counter

# Import your actual modules
from models.encoder import CNNBiLSTMEncoder
from models.decoder import GapDecoder

def load_genome(fasta_path):
    """Load genome from FASTA file"""
    print(f"Loading genome from {fasta_path}...")
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines, join sequence
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    sequence = sequence.upper()
    
    print(f"Loaded {len(sequence):,} bp")
    return sequence

def compute_gc_content(sequence):
    """Calculate G/C percentage"""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)

def create_test_gaps(genome, n_gaps=50, gap_length=50, flank_length=100):
    """
    Create synthetic test gaps from genome
    Returns: list of (left_flank, right_flank, true_gap)
    """
    print(f"\nCreating {n_gaps} test gaps (gap={gap_length}bp, flanks={flank_length}bp)...")
    
    gaps = []
    max_start = len(genome) - (flank_length + gap_length + flank_length)
    
    attempts = 0
    max_attempts = n_gaps * 10
    
    while len(gaps) < n_gaps and attempts < max_attempts:
        attempts += 1
        
        # Random position
        start = random.randint(0, max_start)
        
        left_start = start
        left_end = start + flank_length
        gap_start = left_end
        gap_end = gap_start + gap_length
        right_start = gap_end
        right_end = right_start + flank_length
        
        left_flank = genome[left_start:left_end]
        true_gap = genome[gap_start:gap_end]
        right_flank = genome[right_start:right_end]
        
        # Skip if contains N or non-ACGT
        if all(base in 'ACGT' for base in left_flank + true_gap + right_flank):
            gaps.append((left_flank, right_flank, true_gap))
    
    print(f"Created {len(gaps)} valid test gaps")
    return gaps

def one_hot_encode(sequence):
    """
    One-hot encode DNA sequence
    Returns: tensor of shape [4, length]
    """
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = []
    
    for base in sequence:
        vec = [0, 0, 0, 0]
        if base in base_to_idx:
            vec[base_to_idx[base]] = 1
        encoding.append(vec)
    
    # Transpose to [4, length]
    encoding = list(zip(*encoding))
    return encoding

def predict_gap(encoder, decoder, left_flank, right_flank, gap_length, device):
    """
    Predict gap using trained model with generate() method
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode flanks
        left_encoded = one_hot_encode(left_flank)
        right_encoded = one_hot_encode(right_flank)
        
        # Concatenate left + right
        flanks_encoded = [l + r for l, r in zip(left_encoded, right_encoded)]
        
        # Convert to tensor [1, 4, total_length]
        flanks_tensor = torch.tensor(flanks_encoded, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get context from encoder
        context = encoder(flanks_tensor)  # [1, context_dim]
        
        # Use decoder's generate method (greedy decoding)
        pred_indices = decoder.generate(context, max_len=gap_length, start_token=0)  # [1, gap_length]
        
        # Convert indices to bases
        idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        pred_indices = pred_indices.squeeze(0).cpu().tolist()  # [gap_length]
        predicted_bases = [idx_to_base[idx] for idx in pred_indices]
        
        return ''.join(predicted_bases)

def evaluate_predictions(true_gaps, pred_gaps):
    """
    Compute accuracy metrics
    """
    total_bases = 0
    correct_bases = 0
    base_stats = {b: {'correct': 0, 'total': 0} for b in 'ACGT'}
    
    for true_gap, pred_gap in zip(true_gaps, pred_gaps):
        for true_base, pred_base in zip(true_gap, pred_gap):
            total_bases += 1
            base_stats[true_base]['total'] += 1
            
            if true_base == pred_base:
                correct_bases += 1
                base_stats[true_base]['correct'] += 1
    
    overall_acc = correct_bases / total_bases if total_bases > 0 else 0
    
    per_base_acc = {}
    for base in 'ACGT':
        if base_stats[base]['total'] > 0:
            per_base_acc[base] = base_stats[base]['correct'] / base_stats[base]['total']
        else:
            per_base_acc[base] = 0.0
    
    return {
        'overall_acc': overall_acc,
        'per_base_acc': per_base_acc,
        'base_stats': base_stats,
        'total_bases': total_bases
    }

def analyze_base_distribution(gaps):
    """Analyze base frequency in test gaps"""
    all_bases = ''.join(gaps)
    counter = Counter(all_bases)
    total = len(all_bases)
    
    print("\nBase distribution in test gaps:")
    for base in ['A', 'C', 'G', 'T']:
        count = counter.get(base, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {base}: {pct:.1f}% ({count}/{total})")

def main():
    print("\n" + "="*70)
    print("ZERO-SHOT CROSS-SPECIES EVALUATION")
    print("Testing E. coli-trained model on G/C-rich Streptomyces genome")
    print("="*70 + "\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load trained models - MATCH YOUR TRAINING ARCHITECTURE
    print("Loading trained models...")
    
    encoder = CNNBiLSTMEncoder(
        in_channels=4,
        hidden_channels=128,
        lstm_hidden=128,
        context_dim=256
    )
    
    decoder = GapDecoder(
        context_dim=256,
        hidden_size=256,
        vocab_size=4
    )
    
    try:
        encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
        decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
        print("✓ Models loaded successfully\n")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nMake sure encoder.pth and decoder.pth exist in the current directory.")
        print("Also verify the architecture parameters match your trained models.")
        return
    
    encoder.to(device)
    decoder.to(device)
    
    # Load Streptomyces genome
    genome_file = "data/raw/GCF_000203835.1_ASM20383v1_genomic.fna"
    
    try:
        strep_genome = load_genome(genome_file)
    except FileNotFoundError:
        print(f"Error: {genome_file} not found!")
        print("Make sure the file is in the current directory.")
        return
    
    strep_gc = compute_gc_content(strep_genome)
    print(f"Streptomyces G/C content: {strep_gc:.1%}")
    print(f"(E. coli is typically ~50% G/C)\n")
    
    # Create test gaps
    test_gaps = create_test_gaps(strep_genome, n_gaps=50, gap_length=50, flank_length=100)
    
    if len(test_gaps) < 10:
        print(f"\nWarning: Only created {len(test_gaps)} valid gaps. Results may not be reliable.")
    
    # Analyze base distribution
    true_gaps_only = [gap for _, _, gap in test_gaps]
    analyze_base_distribution(true_gaps_only)
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = []
    
    for i, (left, right, true_gap) in enumerate(test_gaps):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_gaps)}")
        
        try:
            pred_gap = predict_gap(encoder, decoder, left, right, len(true_gap), device)
            predictions.append(pred_gap)
        except Exception as e:
            print(f"\nError predicting gap {i}: {e}")
            predictions.append('A' * len(true_gap))  # Fallback
    
    print("✓ Predictions complete\n")
    
    # Evaluate
    results = evaluate_predictions(true_gaps_only, predictions)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {results['overall_acc']:.1%} ({results['overall_acc']:.4f})")
    print(f"Total bases evaluated: {results['total_bases']}")
    
    print("\nPer-Base Accuracy:")
    for base in ['A', 'C', 'G', 'T']:
        acc = results['per_base_acc'][base]
        stats = results['base_stats'][base]
        print(f"  {base}: {acc:5.1%} ({stats['correct']:4d}/{stats['total']:4d} correct)")
    
    # Analyze prediction distribution
    all_predictions = ''.join(predictions)
    pred_counter = Counter(all_predictions)
    print("\nBase distribution in predictions:")
    for base in ['A', 'C', 'G', 'T']:
        count = pred_counter.get(base, 0)
        pct = count / len(all_predictions) * 100 if len(all_predictions) > 0 else 0
        print(f"  {base}: {pct:.1f}% ({count})")
    
    # Show examples
    print("\n" + "="*70)
    print("Example Predictions (first 10):")
    print("="*70)
    for i in range(min(10, len(test_gaps))):
        _, _, true_gap = test_gaps[i]
        pred_gap = predictions[i]
        
        matches = sum(1 for t, p in zip(true_gap, pred_gap) if t == p)
        acc = matches / len(true_gap)
        
        print(f"\nExample {i}:")
        print(f"TRUE: {true_gap}")
        print(f"PRED: {pred_gap}")
        print(f"Acc:  {acc:.1%} ({matches}/{len(true_gap)} bases)")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"\nYour E. coli test accuracy was ~30-38%")
    print(f"Streptomyces accuracy: {results['overall_acc']:.1%}")
    
    if results['overall_acc'] < 0.25:
        print("\n⚠️  SEVERE DROP: Model likely memorized A/T bias from E. coli")
        print("   G/C-rich sequences completely confuse the model")
        print("\n   Recommended fixes:")
        print("   1. Add class weighting to loss function")
        print("   2. Train on mixed genomes (E. coli + Streptomyces)")
        print("   3. Use data augmentation (random base mutations)")
    elif results['overall_acc'] < 0.30:
        print("\n⚠️  MODERATE DROP: Model shows strong A/T bias")
        print("   Needs class weighting or multi-genome training")
    else:
        print("\n✓  Model generalizes reasonably well across species")
        print("   But still needs improvement overall")
    
    # Check G/C vs A/T accuracy gap
    gc_acc = (results['per_base_acc']['G'] + results['per_base_acc']['C']) / 2
    at_acc = (results['per_base_acc']['A'] + results['per_base_acc']['T']) / 2
    
    if gc_acc < at_acc * 0.5:
        print(f"\n⚠️  G/C accuracy ({gc_acc:.1%}) is less than half of A/T ({at_acc:.1%})")
        print("   Strong evidence of base composition bias")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
