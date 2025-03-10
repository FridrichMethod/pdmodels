#!/bin/bash

#make new directory for model parameters
#e.g.   bash get_model_params.sh "./model_params"

model_params="./model_params/ligandmpnn"

mkdir -p "${model_params}"

#Original ProteinMPNN weights
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt -O "${model_params}/proteinmpnn_v_48_002.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt -O "${model_params}/proteinmpnn_v_48_010.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt -O "${model_params}/proteinmpnn_v_48_020.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt -O "${model_params}/proteinmpnn_v_48_030.pt"

#ProteinMPNN with num_edges=32
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_002.pt -O "${model_params}/proteinmpnn_v_32_002.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_010.pt -O "${model_params}/proteinmpnn_v_32_010.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_020.pt -O "${model_params}/proteinmpnn_v_32_020.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_030.pt -O "${model_params}/proteinmpnn_v_32_030.pt"

#LigandMPNN with num_edges=32; atom_context_num=25
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt -O "${model_params}/ligandmpnn_v_32_005_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt -O "${model_params}/ligandmpnn_v_32_010_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt -O "${model_params}/ligandmpnn_v_32_020_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt -O "${model_params}/ligandmpnn_v_32_030_25.pt"

#LigandMPNN with num_edges=32; atom_context_num=16
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_16.pt -O "${model_params}/ligandmpnn_v_32_005_16.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_16.pt -O "${model_params}/ligandmpnn_v_32_010_16.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_16.pt -O "${model_params}/ligandmpnn_v_32_020_16.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_16.pt -O "${model_params}/ligandmpnn_v_32_030_16.pt"

# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/publication_version_ligandmpnn_v_32_010_25.pt -O "${model_params}/publication_version_ligandmpnn_v_32_010_25.pt"

#Per residue label membrane ProteinMPNN
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/per_residue_label_membrane_mpnn_v_48_020.pt -O "${model_params}/per_residue_label_membrane_mpnn_v_48_020.pt"

#Global label membrane ProteinMPNN
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt -O "${model_params}/global_label_membrane_mpnn_v_48_020.pt"

#SolubleMPNN
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt -O "${model_params}/solublempnn_v_48_002.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt -O "${model_params}/solublempnn_v_48_010.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt -O "${model_params}/solublempnn_v_48_020.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt -O "${model_params}/solublempnn_v_48_030.pt"

#LigandMPNN for side-chain packing (multi-step denoising model)
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt -O "${model_params}/ligandmpnn_sc_v_32_002_16.pt"

# AlphaFold params
if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

ROOT_DIR="./model_params/alphafold/params"
SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
    --directory="${ROOT_DIR}" --preserve-permissions
rm "${ROOT_DIR}/${BASENAME}"
rm "${ROOT_DIR}/LICENSE"
