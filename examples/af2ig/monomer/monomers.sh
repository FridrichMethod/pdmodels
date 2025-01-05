#!/bin/bash

# TensorFlow control
# export TF_FORCE_UNIFIED_MEMORY='1'

# JAX control
# export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'

PYTHONPATH=$(pwd)/../../.. python -m models.af2ig \
    --model_name=model_1_ptm \
    --data_dir=/mnt/d/alphafold \
    --fasta_path=./monomers.fasta \
    --output_dir=./monomers \
    --random_seed=0 \
    --msa_dir=./msas \
    --template_dir=./templates \
    --verbose
