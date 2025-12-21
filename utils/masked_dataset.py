from torch.utils.data import Dataset
import torch
from .encoding import dna_to_indices, MASK_IDX

class MaskedGapDataset(Dataset):
    """
    BERT-style dataset:
    - Input: full sequence with gap positions replaced by MASK_IDX
    - Target: true base indices for every position
    - Loss will be computed only on masked positions.
    """

    def __init__(self, samples):
        """
        samples: list of (left_str, right_str, gap_str)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left, right, gap = self.samples[idx]

        left_ids = dna_to_indices(left)   # [L]
        gap_ids = dna_to_indices(gap)     # [G]
        right_ids = dna_to_indices(right) # [R]

        # Full target sequence
        target = torch.cat([left_ids, gap_ids, right_ids], dim=0)  # [L+G+R]

        # Input: mask the gap region only
        masked = target.clone()
        L = left_ids.shape[0]
        G = gap_ids.shape[0]
        masked[L:L+G] = MASK_IDX

        # Also return a boolean mask for gap positions
        gap_mask = torch.zeros_like(masked, dtype=torch.bool)
        gap_mask[L:L+G] = True

        return masked, target, gap_mask
