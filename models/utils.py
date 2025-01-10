import os
import pickle
import re
import time
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import pymol
import torch
import torch.nn.functional as F
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser, is_aa
from pymol import cmd
from scipy.spatial import distance_matrix
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from models.globals import AA_ALPHABET, AA_DICT, PDB_CHAIN_IDS


def _normalize_submat(submat: np.ndarray) -> np.ndarray:
    """Normalize a substitution matrix to make the sum of the off-diagonal elements equal to 1."""
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


def parse_fasta(fasta_path: str, idx: str = "id", prefix: str = "") -> pd.DataFrame:
    """Parse a fasta file into a pandas DataFrame.

    Args
    ----
    fasta_path: str
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

    with open(fasta_path, "r") as f:
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


def df_to_fasta(df: pd.DataFrame, output_path: str) -> None:
    """Write a DataFrame to a fasta file."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            title = ", ".join(
                f"{key}={value}" for key, value in row.items() if key != "sequence"
            )
            f.write(f">{title}\n{row['sequence']}\n")


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


def calculate_distance_matrix(pdb_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the distance matrix of a PDB file."""
    structure = PDBParser(QUIET=True).get_structure("pdb", pdb_path)
    if len(structure) != 1:
        raise ValueError("Structure contains more than one model.")
    model_ = next(structure.get_models())

    coords_list = []
    mask_list = []
    for chain in model_:
        for residue in chain:
            if not is_aa(residue):
                print(f"Skipping {residue.resname}")
                continue

            mask_list.append(chain.id)
            if residue.resname in {"ALA", "GLY"}:
                coords_list.append(residue["CA"].coord)
            else:
                coords_list.append(residue["CB"].coord)
    coords = np.array(coords_list)
    mask = np.array(mask_list)

    return distance_matrix(coords, coords), mask


def extract_from_af2ig(file_path: str, seqs: str) -> pd.Series:
    """Extract information from an AF2-ig output .pkl file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    plddt = data["plddt"]
    predicted_aligned_error = data["predicted_aligned_error"]
    ptm = data["ptm"].item()
    ranking_confidence = data["ranking_confidence"]

    seq_list = seqs.split(":")
    chain_num = len(seq_list)
    assert sum(len(seq) for seq in seq_list) == len(plddt)

    chain_index = np.array(sum(([i] * len(seq) for i, seq in enumerate(seq_list)), []))
    chain_index_one_hot = np.eye(chain_num, dtype=bool)[chain_index].T

    pkl_dict = {}
    pkl_dict["id"] = os.path.splitext(os.path.basename(file_path))[0].removeprefix(
        "result_"
    )
    pkl_dict["plddt"] = np.mean(plddt)
    pkl_dict["pae"] = np.mean(predicted_aligned_error)
    pkl_dict["ptm"] = ptm
    pkl_dict["ranking_confidence"] = ranking_confidence

    if "iptm" in data:
        iptm = data["iptm"].item()
        pkl_dict["iptm"] = iptm

    for i in range(chain_num):
        chain_id_1 = PDB_CHAIN_IDS[i]
        chain_mask_1 = chain_index_one_hot[i]
        res_num_1 = np.sum(chain_mask_1)
        mean_plddt_1 = np.mean(plddt[chain_mask_1])
        pkl_dict[f"plddt_{chain_id_1}"] = mean_plddt_1.item()

        for j in range(i, chain_num):
            chain_id_2 = PDB_CHAIN_IDS[j]
            chain_mask_2 = chain_index_one_hot[j]
            res_num_2 = np.sum(chain_mask_2)
            pae_interaction_1_2 = (
                np.sum(predicted_aligned_error[chain_mask_1][:, chain_mask_2])
                + np.sum(predicted_aligned_error[chain_mask_2][:, chain_mask_1])
            ) / (2 * res_num_1 * res_num_2)
            pkl_dict[f"pae_interaction_{chain_id_1}_{chain_id_2}"] = (
                pae_interaction_1_2.item()
            )

    return pd.Series(pkl_dict)


def extract_from_esmfold(file_path: str) -> pd.Series:
    """Extract information from an ESMFold output .pt file.

    Can be accelerated with multi-threading.
    """

    data = torch.load(file_path, weights_only=True, map_location="cpu")

    atom37_atom_exists = data["atom37_atom_exists"].cpu().squeeze(0)
    linker_mask = torch.any(atom37_atom_exists, dim=-1)

    atom37_atom_exists = atom37_atom_exists[linker_mask]
    plddt = data["plddt"].cpu().squeeze(0)[linker_mask]
    ptm = data["ptm"].cpu().item()
    predicted_aligned_error = (
        data["predicted_aligned_error"].cpu().squeeze(0)[linker_mask][:, linker_mask]
    )
    mean_plddt = data["mean_plddt"].cpu().item()
    chain_index = data["chain_index"].cpu().squeeze(0)[linker_mask]

    chain_index_one_hot = F.one_hot(chain_index).bool().T
    chain_num = chain_index_one_hot.shape[0]

    pt_dict = {}
    pt_dict["id"] = os.path.splitext(os.path.basename(file_path))[0]
    pt_dict["plddt"] = mean_plddt
    pt_dict["pae"] = torch.mean(predicted_aligned_error).item()
    pt_dict["ptm"] = ptm

    for i in range(chain_num):
        chain_id_1 = PDB_CHAIN_IDS[i]
        chain_mask_1 = chain_index_one_hot[i]
        res_num_1 = torch.sum(chain_mask_1)
        mean_plddt_1 = torch.sum(
            plddt[chain_mask_1] * atom37_atom_exists[chain_mask_1]
        ) / torch.sum(atom37_atom_exists[chain_mask_1])
        pt_dict[f"plddt_{chain_id_1}"] = mean_plddt_1.item()

        for j in range(i, chain_num):
            chain_id_2 = PDB_CHAIN_IDS[j]
            chain_mask_2 = chain_index_one_hot[j]
            res_num_2 = torch.sum(chain_mask_2)
            pae_interaction_1_2 = (
                torch.sum(predicted_aligned_error[chain_mask_1][:, chain_mask_2])
                + torch.sum(predicted_aligned_error[chain_mask_2][:, chain_mask_1])
            ) / (2 * res_num_1 * res_num_2)
            pt_dict[f"pae_interaction_{chain_id_1}_{chain_id_2}"] = (
                pae_interaction_1_2.item()
            )

    return pd.Series(pt_dict)


class PyMOLSession:
    """Context manager for a PyMOL session.

    Example
    -------
    >>> with PyMOLSession():
    >>>     cmd.load("structure.pdb", "structure")
    """

    def __enter__(self):
        cmd.reinitialize()  # Clean up the PyMOL session
        pymol.finish_launching(["pymol", "-cq"])  # Launch PyMOL in headless mode
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cmd.reinitialize()  # Clean up the PyMOL session


def calculate_rmsd(
    mobile: str,
    target: str,
    how: Literal["align", "super"] = "align",
    on: str | tuple[str, str] = "all",
    reports: Sequence[str | tuple[str, str]] = ("all",),
    **kwargs,
) -> list[float]:
    """Calculate the RMSD between the mobile structure and the target structure.

    Args
    ----
    mobile: str
        The path to the mobile structure.
    target: str
        The path to the target structure.
    how: Literal["align", "super"]
        The method to calculate the RMSD. Default is "align".
        align: using a sequence alignment followed by a structural superposition (for sequence similarity > 30%)
        super: using a sequence-independent structure-based superposition (for low sequence similarity)
    on: str | tuple[str, str]
        The selection to align/superimpose the structures. Default is "all".
    reports: Sequence[str | tuple[str, str]]
        The selection to calculate the RMSD. Default is ("all",).

    **kwargs
        Additional keyword arguments for `cmd.align` or `cmd.super`.

    Returns
    -------
    rmsds : list[float]
        The RMSDs between the mobile structure and the target structure for each report selection.

    Notes
    -----
    - This function runs in a PyMOL session, so it cannot be used in multi-threading.
        Use multi-processing instead.
    - If a tuple is provided as selection, the first element is for the mobile structure,
        and the second element is for the target structure.
    """

    def _sele_type_check(sele: str | tuple[str, str]) -> tuple[str, str]:
        if isinstance(sele, str):
            return sele, sele
        elif (
            isinstance(sele, tuple)
            and len(sele) == 2
            and all(isinstance(s, str) for s in sele)
        ):
            return sele
        else:
            raise ValueError(f"Invalid selection: {sele}")

    left_on, right_on = _sele_type_check(on)

    # Set the alignment/superimposition method
    if how == "align":
        func = cmd.align
    elif how == "super":
        func = cmd.super
    else:
        raise ValueError(f"Invalid method: {how}")

    with PyMOLSession():
        # Load the structures
        cmd.load(mobile, "mobile")
        cmd.load(target, "target")

        # Align/superimpose the structures
        func(f"mobile and {left_on}", f"target and {right_on}", **kwargs)

        rmsds = []
        for i, report in enumerate(reports):
            left_report, right_report = _sele_type_check(report)

            # Create the alignment object without touching the structures
            func(
                f"mobile and {left_report}",
                f"target and {right_report}",
                cycles=0,
                transform=0,
                object=f"aln_{i}",
            )
            # Calculate the RMSD between the matched atoms
            rmsd = cmd.rms_cur(
                f"mobile and {left_report} and aln_{i}",
                f"target and {right_report} and aln_{i}",
                matchmaker=-1,
            )
            rmsds.append(rmsd)

    return rmsds


def get_dms_libary(seqs_wt: str) -> list[tuple[str, str]]:
    """Generate a deep mutational scanning library for a given sequence."""
    dms_library = [("wt", seqs_wt)]
    for i, res in enumerate(seqs_wt):
        if res not in AA_ALPHABET:
            continue
        dms_library.extend(
            (f"{res}{i+1}{res_mut}", seqs_wt[:i] + res_mut + seqs_wt[i + 1 :])
            for res_mut in AA_ALPHABET
            if res_mut != res
        )
    return dms_library


class Timer:
    """Context manager to measure the time elapsed in a block of code.

    Example
    -------
    >>> with Timer() as t:
    >>>     time.sleep(1)
    >>> print(t.elapsed)
    """

    def __init__(self):
        self.start = 0
        self.end = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start
