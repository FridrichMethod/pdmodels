#!/bin/bash

# TensorFlow control
# export TF_FORCE_UNIFIED_MEMORY='1'

# JAX control
# export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'
# export JAX_PLATFORMS=cpu

PYTHONPATH=$(pwd)/../../.. python -m pdmodels.af2ig \
    --model_name=model_1_multimer_v3 \
    --data_dir=../../../model_params/alphafold \
    --fasta_path=./multimers.fasta \
    --output_dir=./multimers \
    --random_seed=0 \
    --precalc_dir=./precalc \
    --msa_dir_name=msas \
    --pairing_msa_dir_name=pairing_msas \
    --template_dir_name=templates \
    --verbose \
    # --asynchronous
