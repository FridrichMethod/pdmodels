#!/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)

python -m pdmodels esmfold \
    $ROOT_DIR/assets/monomers/monomers.fasta \
    $ROOT_DIR/assets/monomers/esmfold
