#!/bin/bash

# AlphaFold params

output_dir="$1"

if ! command -v aria2c &>/dev/null; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

ROOT_DIR="${output_dir}/alphafold/params"
SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
    --directory="${ROOT_DIR}" --preserve-permissions
rm "${ROOT_DIR}/${BASENAME}"
rm "${ROOT_DIR}/LICENSE"
