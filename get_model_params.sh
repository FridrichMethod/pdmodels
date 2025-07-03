#!/bin/bash

output_dir="./model_params"

# Create the output directory if it doesn't exist
mkdir -p "${output_dir}"

# Download AlphaFold parameters
bash ./scripts/download_alphafold_weights.sh "${output_dir}"

# Download LigandMPNN parameters
bash ./scripts/download_ligandmpnn_weights.sh "${output_dir}"
