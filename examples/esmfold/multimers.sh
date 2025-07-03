#!/bin/bash

ROOT_DIR=.

python -m pdmodels esmfold \
    $ROOT_DIR/assets/multimers/multimers.fasta \
    $ROOT_DIR/assets/multimers/esmfold
