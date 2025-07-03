#!/bin/bash

export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
# export JAX_PLATFORMS=cpu

PYTHONPATH=$(pwd)/../../.. python -m pdmodels af2ig \
    --model_name=model_1_ptm \
    --data_dir=../../../model_params/alphafold \
    --fasta_path=./monomers.fasta \
    --output_dir=./monomers \
    --random_seed=0 \
    --msa_dir=./msas \
    --template_dir=./templates \
    --verbose
# --asynchronous
