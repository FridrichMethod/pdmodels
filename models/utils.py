import re

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from models.globals import AA_ALPHABET, AA_DICT


def _normalize_submat(submat: np.ndarray) -> np.ndarray:
    assert len(submat.shape) == 2
    assert submat.shape[0] == submat.shape[1] == 20

    diag_sum = np.diag(submat).sum()
    tril_sum = np.tril(submat, k=-1).sum()
    diag_num = submat.shape[0]
    tril_num = diag_num * (diag_num - 1) // 2

    lambda_, mu_ = (
        np.array([diag_num, -diag_sum])
        * tril_num
        / (diag_num * tril_sum - tril_num * diag_sum)
    )
    submat_norm = lambda_ * submat + mu_
    assert np.isclose(np.diag(submat_norm).sum(), 0)
    np.fill_diagonal(submat_norm, 0)
    assert np.isclose(submat_norm.sum(), 380)

    return submat_norm


def count_mutations(
    seqs: list[str], seq0: str, substitution_matrix: str = "identity"
) -> np.ndarray:
    """Count the number of mutations between sequences and a reference sequence of the same length.

    Args
    ----
    seqs: list[str]
        List of sequences to compare.
    seq0: str
        Reference sequence.
    substitution_matrix: str
        Substitution matrix to use. Can be "identity" or a name from Bio.Align.substitution_matrices.load().

    Returns
    -------
    mutation_counts: np.ndarray
        Number of mutations between each sequence and the reference sequence.
    """

    if substitution_matrix == "identity":
        submat: np.ndarray = np.identity(20)
    elif substitution_matrix in substitution_matrices.load():
        submat_fromBio = substitution_matrices.load(substitution_matrix)
        if not set(AA_ALPHABET).issubset(set(submat_fromBio.alphabet)):
            raise ValueError(
                f"Substitution matrix {substitution_matrix} does not contain all 20 amino acids."
            )
        submat: np.ndarray = (
            pd.DataFrame(
                submat_fromBio,
                index=list(submat_fromBio.alphabet),
                columns=list(submat_fromBio.alphabet),
            )
            .loc[list(AA_ALPHABET), list(AA_ALPHABET)]
            .to_numpy()
        )
    else:
        raise ValueError(f"Unknown substitution matrix: {substitution_matrix}")

    submat_norm = _normalize_submat(submat)

    try:
        seqs_idx = np.array([[AA_DICT[aa] for aa in seq] for seq in seqs])
        seq0_idx = np.array([AA_DICT[aa] for aa in seq0])
    except KeyError as e:
        raise KeyError(f"Invalid amino acid: {e}") from None
    except ValueError:
        raise ValueError("Some sequences have different lengths.") from None

    return submat_norm[seqs_idx, seq0_idx].sum(axis=1)


def parse_fasta(fasta_file: str, idx: str = "id", prefix: str = "") -> pd.DataFrame:
    """Parse a fasta file into a pandas DataFrame.

    Args
    ----
    fasta_file: str
        Path to the fasta file.
    idx: str
        Key in the title line to add a prefix.
    prefix: str
        Prefix to add to a specific value of the title key.

    Returns
    -------
    seqs_info: pd.DataFrame
        DataFrame with columns from the title line and the sequence.
    """

    title_regex = r"^>(\w+=[\w\d_.]+)(, \w+=[\w\d_.]+)*$"
    sequence_regex = r"^[ACDEFGHIKLMNPQRSTVWY:]+$"

    with open(fasta_file) as f:
        lines = f.readlines()

    titles = []
    sequences = []
    j = 0
    for i, line in enumerate(lines):
        if re.match(title_regex, line):
            title = line.strip()[1:]
            titles.append(title)
            if i - j:
                raise ValueError(f"Line {line} does not match fasta format")
        elif re.match(sequence_regex, line):
            sequence = line.strip()
            sequences.append(sequence)
            if i - j - 1:
                raise ValueError(f"Line {line} does not match fasta format")
            j = i + 1
        else:
            raise ValueError(f"Line {line} does not match title or sequence regex")

    if len(titles) != len(sequences):
        raise ValueError("Number of titles and sequences do not match")

    title_dicts = []
    for title, sequence in zip(titles, sequences):
        title_dict = {}
        for title_part in title.split(","):
            if title_part:
                key, value = title_part.strip().split("=")
                if key == idx:
                    value = prefix + value
                title_dict[key] = value
        title_dict["sequence"] = sequence
        title_dicts.append(title_dict)

    return pd.DataFrame(title_dicts)


def get_top_percentile(
    df: pd.DataFrame,
    columns: list[str],
    percentile: float = 0.5,
    ascending: bool = True,
    ignore_index=False,
) -> pd.DataFrame:
    """Get top percentile of dataframe based on columns."""

    if ascending:
        df_copy = df[
            (df[columns].rank(method="dense", pct=True) <= percentile).all(axis=1)
        ]
    else:
        df_copy = df[
            (df[columns].rank(method="dense", pct=True) >= percentile).all(axis=1)
        ]

    if ignore_index:
        df = df.reset_index(drop=True)

    return df_copy
