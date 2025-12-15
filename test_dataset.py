from utils.dataset import GapFillDataset

dataset = GapFillDataset(
    fasta_path="data/raw/GCA_000027325.1_ASM2732v1_genomic.fna",
    flank_len=200,
    gap_len=50
)

# Get one sample
left_flank, right_flank, gap = dataset[0]

print("Left flank shape:", left_flank.shape)
print("Right flank shape:", right_flank.shape)
print("Gap shape:", gap.shape)
print("\nLeft flank type:", left_flank.dtype)
print("Gap type:", gap.dtype)
