#!/bin/bash

# export JAX_PLATFORMS=cpu

ROOT_DIR=.

python -m pdmodels af2ig \
    --model_name=model_1_ptm \
    --data_dir=$ROOT_DIR/model_params/alphafold \
    --fasta_path=$ROOT_DIR/assets/monomers/monomers.fasta \
    --output_dir=$ROOT_DIR/assets/monomers/af2ig \
    --random_seed=0 \
    --msa_dir=$ROOT_DIR/assets/monomers/msas \
    --template_dir=$ROOT_DIR/assets/monomers/templates \
    --verbose
# --asynchronous
