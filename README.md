# GeneFill: BERT-Style DNA Gap Filler

[](https://www.python.org/downloads/)
[](https://pytorch.org/get-started/locally/)
[](https://opensource.org/licenses/MIT)

**GeneFill** is a genomic deep learning project designed to reconstruct missing DNA sequences within bacterial genomes. By leveraging a **BERT-style Masked Language Model (MLM)**, GeneFill learns the underlying biological "grammar" of diverse bacterial strains to predict missing bases ($P(\text{gap} \mid \text{flanks})$) using the contextual information from upstream and downstream sequences.

This repository was developed for **BYOP 2026** and includes a complete pipeline for preprocessing, training, and multi-genome evaluation.

-----

## 🧬 Project Overview

The core of GeneFill is a Transformer Encoder that views DNA not just as a string, but as a structured language.

  * **Masked-Token Transformer:** A custom PyTorch implementation of a BERT-style encoder.
  * **Multi-Genome Support:** Pre-configured for 7 major bacterial genomes including *E. coli*, *Shigella*, and *Salmonella*.
  * **Hybrid Training:** Supports training "specialist" models for specific strains or "generalist" models across all included genomes.
  * **Unified Evaluation:** A menu-driven interface to quickly assess accuracy and view real-time gap reconstructions.

-----

## 📂 Repository Structure

```text
GeneFill/
├── data/
│   ├── raw/                # Original FASTA/genome files
│   └── processed/          # Serialized .pkl samples (flank-gap-flank)
├── models/
│   ├── dnamasked_encoder.py # Transformer architecture
│   └── __init__.py
├── utils/
│   ├── encoding.py         # DNA-to-integer mapping
│   └── masked_dataset.py   # PyTorch Dataset & DataLoader logic
├── checkpoints/            # Saved .pth model weights
├── build_samples_all_genomes.py # Data extraction pipeline
├── train_mlm_all_genomes.py     # Training orchestration
├── eval_final.py                # Interactive evaluation suite
└── requirements.txt             # Dependency list
```

-----

## ⚙️ Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/GeneFill.git
    cd GeneFill
    ```

2.  **Environment Setup:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

    *Dependencies include: `torch`, `tqdm`, `numpy`, and `biopython`.*

3.  **Data Preparation:**
    Place your raw genome FASTA files in `data/raw/` ensuring filenames match the configurations in the build scripts.

-----

## 🚀 Workflow

### 1\. Building Training Samples

Extract fixed-length windows (e.g., 200bp flanks and 20bp gaps) and encode them into shared vocabulary IDs: `{A, C, G, T, [MASK], [PAD]}`.

```bash
python build_samples_all_genomes.py
```

### 2\. Training the Model

Train the transformer encoder on the masked language modeling objective. The model learns to predict the identity of `[MASK]` tokens by attending to the surrounding sequence.

```bash
python train_mlm_all_genomes.py
```

*Checkpoints will be saved to the `checkpoints/` directory.*

### 3\. Evaluation

Run the interactive evaluation script to test model performance across the seven supported genomes.

```bash
python eval_final.py
```

**Example Output:**

```text
Choose a genome to evaluate (1–7): 2
Loading E. coli K-12 MG1655...

Accuracy: 92.4%
Ex 01 ✓ | TRUE: TCCTTACCTC | PRED: TCCTTACCTC
Ex 02 ✗ | TRUE: TAAGTACTGT | PRED: TAAGTACTTT
```

-----

## 🧠 Model Architecture

The GeneFill architecture utilizes a high-capacity Transformer Encoder:

  * **Input Layer:** DNA bases $\rightarrow$ Learnable Embeddings + Sinusoidal Positional Encodings (supports windows up to 700bp).
  * **Encoder Blocks:** Multi-head self-attention layers with LayerNorm and Dropout for regularization.
  * **Objective:** Cross-entropy loss computed exclusively on masked gap positions.

-----

## 📝 Reproducibility & Notes

  * **Consistency:** Ensure `NUCLEOTIDES` and `PAD_IDX` mappings remain identical across all scripts to prevent encoding mismatches.
  * **Hardware:** Training is optimized for CUDA; however, evaluation can run efficiently on CPU.
  * **Versions:** Developed using PyTorch 2.x and Python 3.9.

-----

## 📚 Acknowledgments & Citations

If you use this codebase, please cite the following foundational works:

  * *Vaswani, A., et al. (2017).* **"Attention Is All You Need"** (Transformer Architecture).
  * *Dalla-Torre, H., et al. (2023).* **"The Nucleotide Transformer"** (Genomic Language Models).
  * *Additional references:* DLGapCloser and Gene-LLMs.

-----

*Developed for the 2026 BYOP Project.*
