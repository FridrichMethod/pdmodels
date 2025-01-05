import contextlib

with contextlib.suppress(ImportError):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False

from models.globals import *

with contextlib.suppress(ImportError):
    from models.af2ig import *

with contextlib.suppress(ImportError):
    from models.esmfold import *

with contextlib.suppress(ImportError):
    from models.esm2 import *
    from models.esmif import *
    from models.mpnn import *


__all__ = [
    "Af2Ig",
    "ESMFold",
    "ESM2",
    "ESMIF",
    "MPNN",
]
