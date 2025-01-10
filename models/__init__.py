import torch

from models.af2ig import Af2Ig
from models.esm2 import ESM2
from models.esmfold import ESMFold
from models.esmif import ESMIF
from models.mpnn import MPNN

torch.backends.cuda.matmul.allow_tf32 = False

__all__ = ["Af2Ig", "ESMFold", "ESM2", "ESMIF", "MPNN"]
