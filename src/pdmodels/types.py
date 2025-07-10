from typing import TypedDict

import torch
from torch._prims_common import DeviceLikeType

Device = DeviceLikeType | None


class ScoreDict(TypedDict):
    """Type definition for a score dictionary."""

    entropy: torch.Tensor
    target: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor
