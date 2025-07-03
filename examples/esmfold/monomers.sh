#!/bin/bash

ROOT_DIR=.

python -m pdmodels esmfold \
    $ROOT_DIR/assets/monomers/monomers.fasta \
    $ROOT_DIR/assets/monomers/esmfold
