# eval_final.py - Menu-based evaluator for 7 genomes

import pickle
import torch
from torch.utils.data import DataLoader

from models import DNAMaskedEncoder
from utils.masked_dataset import MaskedGapDataset
from utils.encoding import PAD_IDX, NUCLEOTIDES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EXAMPLES_TO_PRINT = 10


GENOMES = {
    "1": {
        "name": "ECOR (diverse E. coli reference strains)",
        "model": "checkpoints/ecor.pth",
        "data": "data/processed/ecor_gapfill_samples.pkl",
    },
    "2": {
        "name": "E. coli K-12 MG1655",
        "model": "checkpoints/k12.pth",
        "data": "data/processed/k12_gapfill_samples.pkl",
    },
    "3": {
        "name": "UTI89 (uropathogenic E. coli)",
        "model": "checkpoints/uti89.pth",
        "data": "data/processed/uti89_gapfill_samples.pkl",
    },
    "4": {
        "name": "Shigella flexneri",
        "model": "checkpoints/shigella.pth",
        "data": "data/processed/shigella_gapfill_samples.pkl",
    },
    "5": {
        "name": "Enterobacter cloacae",
        "model": "checkpoints/enterobacter.pth",
        "data": "data/processed/enterobacter_gapfill_samples.pkl",
    },
    "6": {
        "name": "Klebsiella pneumoniae",
        "model": "checkpoints/klebsiella.pth",
        "data": "data/processed/klebsiella_gapfill_samples.pkl",
    },
    "7": {
        "name": "Salmonella enterica Typhimurium LT2",
        "model": "checkpoints/salmonella.pth",
        "data": "data/processed/salmonella_gapfill_samples.pkl",
    },
}


def collate_fn(batch):
    masked_list, target_list, mask_list = zip(*batch)
    lengths = [x.shape[0] for x in masked_list]
    max_len = max(lengths)

    padded_masked, padded_target, padded_gapmask = [], [], []

    for masked, target, gmask in zip(masked_list, target_list, mask_list):
        pad_len = max_len - masked.shape[0]
        if pad_len > 0:
            pad_val = PAD_IDX
            masked = torch.cat(
                [masked, torch.full((pad_len,), pad_val, dtype=torch.long)],
                dim=0
            )
            target = torch.cat(
                [target, torch.full((pad_len,), pad_val, dtype=torch.long)],
                dim=0
            )
            gmask = torch.cat(
                [gmask, torch.zeros(pad_len, dtype=torch.bool)],
                dim=0
            )
        padded_masked.append(masked)
        padded_target.append(target)
        padded_gapmask.append(gmask)

    return (
        torch.stack(padded_masked, dim=0),
        torch.stack(padded_target, dim=0),
        torch.stack(padded_gapmask, dim=0),
    )


def indices_to_dna(id_tensor):
    return "".join(NUCLEOTIDES[i] for i in id_tensor.tolist())


def choose_genome():
    print("=" * 70)
    print("GENOME SELECTION MENU")
    print("=" * 70)
    for key, info in GENOMES.items():
        print(f"{key}. {info['name']}")
    print("=" * 70)

    choice = None
    while choice not in GENOMES:
        choice = input("Choose a genome to evaluate (1–7): ").strip()
        if choice not in GENOMES:
            print("Invalid choice. Please enter a number from 1 to 7.\n")

    return GENOMES[choice]


def main():
    selection = choose_genome()
    MODEL_PATH = selection["model"]
    DATA_PATH = selection["data"]

    print()
    print("=" * 70)
    print(f"EVALUATION: BERT-STYLE MASKED DNA MODEL")
    print("=" * 70)
    print(f"Genome: {selection['name']}")
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Data:   {DATA_PATH}")
    print("=" * 70)
    print()

    # Load samples
    with open(DATA_PATH, "rb") as f:
        samples = pickle.load(f)

    print(f"Total samples loaded: {len(samples):,}")

    dataset = MaskedGapDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load model
    model = DNAMaskedEncoder().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    print()
    print("Evaluating...")
    print("-" * 70)

    total_correct = 0
    total_mask = 0
    printed = 0

    with torch.no_grad():
        for masked_ids, targets, gap_mask in loader:
            masked_ids = masked_ids.to(DEVICE)
            targets = targets.to(DEVICE)
            gap_mask = gap_mask.to(DEVICE)

            pad_mask = (masked_ids == PAD_IDX)
            logits = model(masked_ids, pad_mask)      # [B,L,4]
            preds = logits.argmax(dim=-1)             # [B,L]

            correct = ((preds == targets) & gap_mask).sum().item()
            n_mask = gap_mask.sum().item()

            total_correct += correct
            total_mask += n_mask

            # Print a few example gaps (compact)
            if printed < NUM_EXAMPLES_TO_PRINT:
                for b in range(masked_ids.size(0)):
                    if printed >= NUM_EXAMPLES_TO_PRINT:
                        break

                    mask_pos = gap_mask[b].nonzero(as_tuple=False).squeeze(1)
                    if mask_pos.numel() == 0:
                        continue

                    true_gap = targets[b, mask_pos].cpu()
                    pred_gap = preds[b, mask_pos].cpu()

                    true_str = indices_to_dna(true_gap)
                    pred_str = indices_to_dna(pred_gap)

                    match = "✓" if true_str == pred_str else "✗"
                    print(
                        f"Ex {printed+1:02d} {match} | "
                        f"TRUE: {true_str} | PRED: {pred_str}"
                    )

                    printed += 1

    acc = total_correct / max(total_mask, 1)
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Genome:              {selection['name']}")
    print(f"Masked-token acc.:   {acc:.4f}")
    print(f"Total masked tokens: {total_mask:,}")
    print(f"Correct predictions: {total_correct:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
