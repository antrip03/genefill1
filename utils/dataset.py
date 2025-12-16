from torch.utils.data import Dataset
import torch
from .encoding import dna_to_one_hot, NUCLEOTIDES

BASE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}

class GapFillDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of (left_flank_str, right_flank_str, gap_str)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

  
    def __getitem__(self, idx):
        left, right, gap = self.samples[idx]



        left_oh = dna_to_one_hot(left)     # [flank_len, 4]
        right_oh = dna_to_one_hot(right)   # [flank_len, 4]

        gap_idx = torch.tensor(
            [BASE_TO_INDEX[b] for b in gap],
            dtype=torch.long
        )                                   # [gap_len]

        return left_oh, right_oh, gap_idx
