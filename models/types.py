from typing import TypedDict

import torch


class ScoreDict(TypedDict):
    """Type definition for a score dictionary."""

    entropy: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor
