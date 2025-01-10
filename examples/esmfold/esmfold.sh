#!/bin/bash

PYTHONPATH=$(pwd)/../.. python -m models.esmfold ./multimers.fasta ./output
