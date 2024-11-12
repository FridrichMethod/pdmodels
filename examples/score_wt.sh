#!/bin/bash

PYTHONPATH=$(pwd)/.. python -m models.score \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "../model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "$1" \
    --out_folder "$2" \
    --chains_to_design "A" \
    --use_sequence 1 \
    --autoregressive_score 0 \
    --single_aa_score 1 \
    --ligand_mpnn_use_side_chain_context 1 \
    --batch_size 16 \
    --number_of_batches 1 \
    --verbose 1

    # --checkpoint_ligand_mpnn "../model_params/ligandmpnn_v_32_020_25.pt" \