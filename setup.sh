#!/bin/bash

# This script sets up the environment for `pdmodels`

# Create a conda environment for `pdmodels`
# conda create -n pdmodels python=3.12 -y
# conda activate pdmodels

# Enable AlphaFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Set up conda environment variables for OpenFold
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set up CUDA environment variables
export CUDA_HOME=$CONDA_PREFIX
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH

# Set up conda environment variables for AlphaFold3
mv "$CONDA_PREFIX"/compiler_compat/ld{,.bak}

# Install CUDA toolkit matching PyTorch (cu128)
conda install -c nvidia cuda-toolkit=12.8 -y

# Install uv and set up Python 3.12
pip install uv

# Install PyMOL
conda install -c conda-forge glew=2.1.0 -y
conda install -c schrodinger vtk-m=1.8.0 -y
conda install -c conda-forge -c schrodinger pymol-bundle -y

# Install multiple sequence alignment tools
conda install -c conda-forge -c bioconda kalign hhsuite hmmer blast mmseqs2 -y

# Install AlphaFold
uv pip install git+https://github.com/FridrichMethod/alphafold.git@dev

# Install AlphaFold3
conda install -c conda-forge pdbfixer -y
uv pip install git+https://github.com/FridrichMethod/alphafold3.git@dev

# Install PyTorch Geometric and its dependencies
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install and patch ESM2
uv pip install git+https://github.com/facebookresearch/esm.git
patch "$(python -c 'import site,sys; sys.stdout.write(next(p for p in site.getsitepackages() if "site-packages" in p) + "/esm/inverse_folding/util.py")')" <<'EOF'
@@ -12,1 +12,1 @@
-from biotite.structure import filter_backbone
+from biotite.structure import filter_peptide_backbone as filter_backbone
EOF

# Install OpenFold
uv pip install git+https://github.com/aqlaboratory/openfold.git --no-build-isolation

# Install the pdmodels package
uv pip install -e .[dev,mypy]
