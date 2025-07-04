"""Module to store global variables."""

import subprocess

# Global variables
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_DICT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ROOT_DIR = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()
