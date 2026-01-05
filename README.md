# Genefill
Project for BYOP 2026

GeneFill: BERT‑Style DNA Gap Filler
A PyTorch implementation of a BERT‑style masked language model for DNA sequences that predicts missing bases given flanking context across multiple bacterial genomes. The project includes data preprocessing, model training, and a unified evaluation interface for seven genomes.
​

1. Project overview

This repository implements:

A masked‑token transformer encoder for DNA sequences (BERT‑style MLM).

A preprocessing pipeline that extracts flank–gap–flank windows and builds training samples for multiple genomes.

Training scripts that can either train per‑genome specialists or a combined model over all genomes.

Evaluation scripts that report masked‑token accuracy and print example gap reconstructions.
​

Target use‑case: filling gaps in bacterial genome assemblies (e.g., E. coli, Shigella, Klebsiella, Salmonella) using a learned probabilistic model 
P
(
gap
∣
flanks
)
P(gap∣flanks).
​

2. Repository structure
A typical layout for this project:

data/raw/ – Original FASTA/genome files per genome.

data/processed/ – Serialized sample files (*_gapfill_samples.pkl) produced by the build script.

models/ – Model definitions (e.g., dnamasked_encoder.py / __init__.py).

utils/ – Helper code (encoding.py, masked_dataset.py, etc.).

checkpoints/ – Saved model weights (*.pth) per genome or combined model.

build_samples_all_genomes.py – Build masked gap‑filling samples for all genomes.

train_mlm_all_genomes.py – Train the masked language model across all genomes.

eval_final.py – Menu‑driven evaluation script over seven genomes.

​

3. Setup and installation
Clone the repository and create a Python environment (recommended: Python 3.9+).
​

Install dependencies, for example:

bash
pip install -r requirements.txt
At minimum you need: torch, tqdm, numpy, and any packages used in requirements.txt.
​

Ensure your raw genome data is placed in data/raw/ or match the expected paths inside build_samples_all_genomes.py.
​

4. Building training samples
build_samples_all_genomes.py reads raw genomes, extracts windows, and creates pickled datasets for each genome.
​

Command
bash
python build_samples_all_genomes.py
What it does
Parses each genome (e.g., ECOR, K‑12, UTI89, Shigella, Enterobacter cloacae, Klebsiella pneumoniae, Salmonella Typhimurium).
​

Extracts fixed‑length flank–gap–flank windows (e.g., flank length 100–300, gap length 10–50, depending on your settings).
​

Encodes DNA into integer IDs with a shared vocabulary: {A, C, G, T, [MASK], [PAD]}.
​

Saves per‑genome sample files such as:

data/processed/ecor_gapfill_samples.pkl

data/processed/k12_gapfill_samples.pkl

data/processed/uti89_gapfill_samples.pkl

etc.
​

If you change flank/gap lengths or sampling strategies, update the arguments or constants in build_samples_all_genomes.py.
​

5. Training the masked language model
train_mlm_all_genomes.py trains a BERT‑style transformer encoder with a masked language modeling objective over the combined samples.
​

Command
bash
python train_mlm_all_genomes.py​
Saves checkpoints into checkpoints/, e.g.:

text
checkpoints/ecor.pth
checkpoints/k12.pth
...
checkpoints/joint_all_genomes.pth
(Adjust names according to how you configured train_mlm_all_genomes.py.)
​

6. Evaluating models (menu over 7 genomes)
eval_final.py provides a simple text menu to evaluate any of the seven genomes with a single script.
​

Interactive usage (terminal)
From a terminal:

bash
python eval_final.py
You will see a menu like:
1. ECOR (diverse E. coli reference strains)
2. E. coli K-12 MG1655
3. UTI89 (uropathogenic E. coli)
4. Shigella flexneri
5. Enterobacter cloacae
6. Klebsiella pneumoniae
7. Salmonella enterica Typhimurium LT2
======================================================================
Choose a genome to evaluate (1–7):
Enter a number 1–7 to select the genome. The script then:
​

Loads the corresponding model checkpoint and sample file (paths defined in the GENOMES dictionary inside eval_final.py).
​

Runs masked‑token evaluation, computing:

Total masked tokens

Number of correct predictions

Masked‑token accuracy.
​

Prints a few example gaps in compact form:

text
Ex 01 ✓ | TRUE: TCCTTACCTC | PRED: TCCTTACCTC
Ex 02 ✗ | TRUE: TAAGTACTGT | PRED: TAAGTACTTT
...
Non‑interactive usage (e.g., Jupyter)
In environments that do not support input(), you can modify eval_final.py to accept a command‑line argument (e.g., python eval_final.py 3 to evaluate UTI89) or hard‑code the choice; see the script comments for details.
​

7. Model architecture (high level)
Input encoding

DNA bases mapped to integer IDs with special mask/pad tokens.

Positional embeddings added up to a maximum window length (e.g., 700).
​

Transformer encoder

Several self‑attention layers (multi‑head attention + feed‑forward).

Layer normalization and dropout for stability.
​

Output and loss

Linear classifier over {A, C, G, T} at each position.

Cross‑entropy loss computed only on masked gap positions.
​

This setup allows the model to see both left and right flanks while reconstructing the gap, matching the biological setting where flanks are known and the interior needs to be imputed.
​

8. Reproducibility notes
Fix random seeds in train_mlm_all_genomes.py if you want deterministic runs across devices.
​

Checkpoint formats may depend on the PyTorch version: if loading fails, ensure your PyTorch version matches the one used for training or adapt the loading code accordingly.
​

Keep NUCLEOTIDES, PAD_IDX, and token mappings consistent across build, train, and eval scripts.
​

9. How to cite / acknowledge
If you use this codebase in a report or project presentation, consider citing:

The original transformer paper (“Attention Is All You Need”) for the architecture.

Relevant genomic language‑model works (e.g., DLGapCloser, Nucleotide Transformer, GENA‑LM, Gene‑LLMs) as described in your bibliography
