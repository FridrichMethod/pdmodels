import os

import torch

from pdmodels.af2ig import Af2Ig
from pdmodels.esm2 import ESM2
from pdmodels.esmfold import ESMFold
from pdmodels.esmif import ESMIF
from pdmodels.mpnn import MPNN
from pdmodels.revor import ReVor

__version__ = "0.1.0"
__author__ = "Zhaoyang Li"

__all__ = ["Af2Ig", "ESMFold", "ESM2", "ESMIF", "MPNN", "ReVor"]

# Set PyTorch backend options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable AlphaFold for WSL2
os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "4.0")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Set up conda environment variables for OpenFold
if conda_prefix := os.environ.get("CONDA_PREFIX"):
    libdir = os.path.join(conda_prefix, "lib")
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if libdir not in paths:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        os.environ["LIBRARY_PATH"] = f"{libdir}:{os.environ.get('LIBRARY_PATH', '')}"
