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
    """Count the number of mutations between sequences and a reference sequence of the same length."""
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
