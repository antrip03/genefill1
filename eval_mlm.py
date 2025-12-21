# eval_mlm.py - Evaluate BERT-style masked DNA model

import pickle
import torch
from torch.utils.data import DataLoader

from models import DNAMaskedEncoder
from utils.masked_dataset import MaskedGapDataset
from utils.encoding import PAD_IDX, NUCLEOTIDES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "dna_mlm_transformer.pth"
DATA_PATH = "data/processed/lambda_gapfill_samples.pkl"  # updated for Lambda Phage
BATCH_SIZE = 64
NUM_EXAMPLES_TO_PRINT = 10


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
    # id_tensor: [L] with values 0..3
    return "".join(NUCLEOTIDES[i] for i in id_tensor.tolist())


def main():
    print("=" * 70)
    print("EVALUATION: BERT-STYLE MASKED DNA MODEL")
    print("=" * 70)
    print(f"Device:  {DEVICE}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Data:    {DATA_PATH}")
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
        collate_fn=collate_fn
    )

    # Load model
    model = DNAMaskedEncoder().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    print()

    total_correct = 0
    total_mask = 0
    printed = 0

    print("Evaluating...")
    print("-" * 70)

    with torch.no_grad():
        for batch_idx, (masked_ids, targets, gap_mask) in enumerate(loader):
            masked_ids = masked_ids.to(DEVICE)
            targets = targets.to(DEVICE)
            gap_mask = gap_mask.to(DEVICE)

            pad_mask = (masked_ids == PAD_IDX)
            logits = model(masked_ids, pad_mask)          # [B,L,4]
            preds = logits.argmax(dim=-1)                 # [B,L]

            # Accuracy only on gap positions
            correct = ((preds == targets) & gap_mask).sum().item()
            n_mask = gap_mask.sum().item()

            total_correct += correct
            total_mask += n_mask

            # Print a few example gaps
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
                    print(f"Example {printed+1} {match}")
                    print(f"  TRUE: {true_str}")
                    print(f"  PRED: {pred_str}")
                    print()

                    printed += 1

    acc = total_correct / max(total_mask, 1)
    print("=" * 70)
    print(f"RESULTS")
    print("=" * 70)
    print(f"Masked-token accuracy: {acc:.4f}")
    print(f"Total masked tokens:   {total_mask:,}")
    print(f"Correct predictions:   {total_correct:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()

