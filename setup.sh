#!/bin/bash

# This script sets up the environment for `pdmodels`

# Create a conda environment for `pdmodels`
# conda create -n pdmodels python=3.12 --yes
# conda activate pdmodels

# Enable AlphaFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Set up conda environment variables for OpenFold
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install uv and set up Python 3.12
pip install uv

# Install Jupyter and PyMOL
conda install -c conda-forge jupyter --yes
conda install -c schrodinger -c conda-forge pymol --yes
# Update SQLite to fix issue with Jupyter after installing PyMOL
conda update -c conda-forge sqlite --yes

# Install AlphaFold and AlphaFold3
uv pip install git+https://github.com/google-deepmind/alphafold.git
uv pip install git+https://github.com/google-deepmind/alphafold3.git

# Update jax and related libraries
uv pip install jax[cuda12] --upgrade
uv pip install dm-haiku --upgrade
uv pip install triton --upgrade
uv pip install jax-triton --upgrade
uv pip install flax --upgrade
uv pip install deepspeed --upgrade

# Install PyTorch Geometric and its dependencies
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Install and patch ESM2
uv pip install git+https://github.com/facebookresearch/esm.git
patch "$(python -c 'import site,sys; sys.stdout.write(next(p for p in site.getsitepackages() if "site-packages" in p) + "/esm/inverse_folding/util.py")')" <<'EOF'
@@ -12,1 +12,1 @@
-from biotite.structure import filter_backbone
+from biotite.structure import filter_peptide_backbone as filter_backbone
EOF

# Install OpenFold
uv pip install git+https://github.com/aqlaboratory/openfold.git --no-build-isolation

# Update all CUDA libraries
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade
uv pip install nvidia-cudnn-cu12 --upgrade
# pip list --format=freeze |
#     cut -d'=' -f1 |
#     grep '^nvidia-' |
#     xargs uv pip install --upgrade

# Install ProDy
uv pip install git+https://github.com/prody/ProDy.git
uv pip install numpy --upgrade

# Install additional libraries
uv pip install transformers
uv pip install scikit-learn
uv pip install rdkit
uv pip install matplotlib
uv pip install seaborn
uv pip install ipympl
uv pip install py3Dmol
uv pip install biotite
uv pip install mdtraj
uv pip install modelcif

# Install the pdmodels package
uv pip install -e .
