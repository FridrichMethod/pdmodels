#!/bin/bash

PYTHONPATH=$(pwd)/.. python -m pdmodels.run \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "../model_params/ligandmpnn_v_32_020_25.pt" \
    --temperature 0.3 \
    --pdb_path "$1" \
    --out_folder "$2" \
    --redesigned_residues "A1 A3 A4 A5 A7 A8 A9 A13 A14 A15 A19 A20 A21 A23 A24 A25 A26 A27 A39 A41 A44 A45 A46 A48 A50 A52 A53 A67 A68 A69 A72 A73 A74 A75 A76 A77 A78 A79 A80 A81 A82 A83 A84 A85 A86 A88 A89 A91 A92 A93 A95 A97 A99 A100 A102 A114 A116 A118 A119 A120 A121 A123 A124" \
    --omit_AA "C" \
    --ligand_mpnn_use_side_chain_context 1 \
    --save_stats 1 \
    --zero_indexed 1 \
    --fasta_seq_separation ":" \
    --batch_size 256 \
    --number_of_batches 256 \
    --verbose 1

# --checkpoint_ligand_mpnn "../model_params/ligandmpnn_v_32_030_25.pt" \
# --temperature 0.2 \
