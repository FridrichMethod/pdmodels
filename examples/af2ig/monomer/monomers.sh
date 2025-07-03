#!/bin/bash

# export JAX_PLATFORMS=cpu

python -m pdmodels af2ig \
    --model_name=model_1_ptm \
    --data_dir=../../../model_params/alphafold \
    --fasta_path=./monomers.fasta \
    --output_dir=./monomers \
    --random_seed=0 \
    --msa_dir=./msas \
    --template_dir=./templates \
    --verbose
# --asynchronous
