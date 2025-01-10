"""Module to store global variables."""

from typing import TypedDict

import torch


class ScoreDict(TypedDict):
    """Type definition for a score dictionary."""

    entropy: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor


# Global variables
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_DICT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
