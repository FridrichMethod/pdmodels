#!/bin/bash

# export JAX_PLATFORMS=cpu

ROOT_DIR=.

python -m pdmodels af2ig \
    --model_name=model_1_multimer_v3 \
    --data_dir=$ROOT_DIR/model_params/alphafold \
    --fasta_path=$ROOT_DIR/assets/multimers/multimers.fasta \
    --output_dir=$ROOT_DIR/assets/multimers/af2ig \
    --random_seed=0 \
    --precalc_dir=$ROOT_DIR/assets/multimers/precalc \
    --msa_dir_name=msas \
    --pairing_msa_dir_name=pairing_msas \
    --template_dir_name=templates \
    --verbose
# --asynchronous
