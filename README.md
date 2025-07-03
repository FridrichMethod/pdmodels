# Protein Design Models

A unified library for Alphafold2, ESM2, ProteinMPNN, and more.

---

```python
pdmodels/
│
├── af2ig.py  # AlphaFold 2 and AlphaFold-Multimer implementation with initial guess of MSA and templates
├── esmfold.py  # ESMFold implementation
├── esm2.py  # ESM2 with batch single sequence scoring function
├── esmif.py  # ESM-IF batch version
├── mpnn.py  # ProteinMPNN and LigandMPNN batch version
└── revor.py  # Reversed evolution using inverse folding models
```

## Installation

```bash
# Clone the repository
git clone https://github.com/FridrichMethod/pdmodels.git
cd pdmodels

# Create a conda environment
# Make sure you have conda installed, e.g., via Miniconda or Anaconda
conda create -n pdmodels python=3.12 --yes
conda activate pdmodels

# Install the required packages
bash setup.sh
```
