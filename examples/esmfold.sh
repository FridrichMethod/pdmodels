#!/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)

python -m pdmodels esmfold \
    $ROOT_DIR/assets/multimers/multimers.fasta \
    $ROOT_DIR/assets/multimers/esmfold
