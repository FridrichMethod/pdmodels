import argparse
import copy
import os
import pickle
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, MutableMapping

import jax.numpy as jnp
import numpy as np
from absl import logging
from alphafold.common import protein, residue_constants
from alphafold.data import (
    feature_processing,
    parsers,
    pipeline,
    pipeline_multimer,
    templates,
)
from alphafold.data.tools import hhsearch, kalign
from alphafold.model import config, data, model
from Bio.PDB import PDBParser, is_aa
from Bio.SeqIO.FastaIO import SimpleFastaParser
from tqdm.auto import tqdm

from pdmodels.basemodels import BaseModel
from pdmodels.utils import Timer

GeneralFeatureDict = MutableMapping[str, list[Any]]


def _jnp_to_np(output: dict[str, Any]) -> dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def run_msa_tool(
    msa_path: str,
    msa_format: str,
    max_sto_sequences: int | None = None,
) -> Mapping[str, Any]:
    """Retrieves MSA from file."""
    logging.info("Reading MSA from file %s", msa_path)
    if msa_format == "sto" and max_sto_sequences is not None:
        precomputed_msa = parsers.truncate_stockholm_msa(msa_path, max_sto_sequences)
        result = {"sto": precomputed_msa}
    else:
        with open(msa_path, "r") as f:
            result = {msa_format: f.read()}
    return result


def _make_msa_features_from_files(
    msa_dir: str,
    uniref_max_hits: int = 10000,
    mgnify_max_hits: int = 501,
) -> pipeline.FeatureDict:
    """Constructs a feature dict of MSA features from files.

    The MSA files are expected to be the same as AlphaFold MSA tools output.
    """

    logging.info("Making MSA features...")

    if not msa_dir:
        logging.info("No MSA provided.")
        return {}
    if not os.path.isdir(msa_dir):
        raise ValueError(f"MSA directory {msa_dir} not found.")

    jackhmmer_uniref90_result = run_msa_tool(
        msa_path=os.path.join(msa_dir, "uniref90_hits.sto"),
        msa_format="sto",
        max_sto_sequences=uniref_max_hits,
    )
    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result["sto"])
    jackhmmer_mgnify_result = run_msa_tool(
        msa_path=os.path.join(msa_dir, "mgnify_hits.sto"),
        msa_format="sto",
        max_sto_sequences=mgnify_max_hits,
    )
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result["sto"])
    if os.path.exists(
        hhblits_bfd_uniref_path := os.path.join(msa_dir, "bfd_uniref_hits.a3m")
    ):
        hhblits_bfd_uniref_result = run_msa_tool(
            msa_path=hhblits_bfd_uniref_path,
            msa_format="a3m",
        )
        bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result["a3m"])
    elif os.path.exists(
        jackhmmer_small_bfd_path := os.path.join(msa_dir, "small_bfd_hits.sto")
    ):
        jackhmmer_small_bfd_result = run_msa_tool(
            msa_path=jackhmmer_small_bfd_path,
            msa_format="sto",
        )
        bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result["sto"])
    else:
        raise ValueError("Could not find BFD MSA.")

    msa_features_for_all = pipeline.make_msa_features(
        (uniref90_msa, bfd_msa, mgnify_msa)
    )
    for key, value in msa_features_for_all.items():
        logging.info("%s: %s", key, value.shape)

    return msa_features_for_all


def parse_pdb(pdb_path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Parses a PDB file to extract atom positions, atom masks and sequence."""
    logging.info("Reading PDB from file %s", pdb_path)

    structure = PDBParser(QUIET=True).get_structure("pdb", pdb_path)

    if len(structure) != 1:
        raise ValueError("Structure contains more than one model.")
    model_ = next(structure.get_models())
    logging.info("Use model %s", model_.id)
    if len(model_) != 1:
        raise ValueError("Model contains more than one chain.")
    chain = next(model_.get_chains())
    logging.info("Use chain %s", chain.id)

    num_res = len([res for res in chain if is_aa(res)])
    logging.info("Number of residues: %s", num_res)
    all_atoms_positions = np.zeros((num_res, residue_constants.atom_type_num, 3))
    all_atoms_mask = np.zeros(
        (num_res, residue_constants.atom_type_num), dtype=np.int32
    )
    seq = []

    for i, residue in enumerate(chain):
        if not is_aa(residue):
            continue
        if residue.resname in residue_constants.restype_3to1:
            seq.append(residue_constants.restype_3to1[residue.resname])
        elif residue.resname == "MSE":
            logging.info("MSE %s is treated as MET in AlphaFold", residue.id[1])
            seq.append("M")
        elif residue.resname == "SEC":
            logging.info("SEC %s is treated as CYS in AlphaFold", residue.id[1])
            seq.append("C")
        else:
            logging.info(
                "Unknown residue %s %s in PDB, using X instead",
                residue.resname,
                residue.id[1],
            )
            seq.append("X")

        for atom in residue:
            if atom.name in residue_constants.atom_order:
                atom_index = residue_constants.atom_order[atom.name]
            elif atom.name == "SE":
                # Put the coordinates of the selenium atom in the sulphur column
                atom_index = residue_constants.atom_order["SD"]
            else:
                logging.info(
                    "Unknown atom %s in residue %s %s",
                    atom.name,
                    residue.resname,
                    residue.id[1],
                )
                continue
            all_atoms_positions[i, atom_index] = atom.coord
            all_atoms_mask[i, atom_index] = 1

        # Fix naming errors in arginine residues where NH2 is incorrectly
        # assigned to be closer to CD than NH1
        cd = residue_constants.atom_order["CD"]
        nh1 = residue_constants.atom_order["NH1"]
        nh2 = residue_constants.atom_order["NH2"]
        if residue.resname == "ARG" and np.all(
            all_atoms_mask[[i, i, i], [cd, nh1, nh2]]
        ):
            dist_nh1_cd = np.linalg.norm(
                all_atoms_positions[i, nh1] - all_atoms_positions[i, cd]
            )
            dist_nh2_cd = np.linalg.norm(
                all_atoms_positions[i, nh2] - all_atoms_positions[i, cd]
            )
            if dist_nh2_cd < dist_nh1_cd:
                all_atoms_positions[i, nh1], all_atoms_positions[i, nh2] = (
                    all_atoms_positions[i, nh2].copy(),
                    all_atoms_positions[i, nh1].copy(),
                )
                all_atoms_mask[i, nh1], all_atoms_mask[i, nh2] = (
                    all_atoms_mask[i, nh2].copy(),
                    all_atoms_mask[i, nh1].copy(),
                )

    sequence = "".join(seq)
    logging.info("Sequence: %s", sequence)

    return all_atoms_positions, all_atoms_mask, sequence


def _msa_to_indices_hit(msa: parsers.Msa) -> list[np.ndarray]:
    """Converts an MSA object to a list of hit indices."""
    query_sequence, *hit_sequences = msa.sequences
    query_mask = np.array([res != "-" for res in query_sequence])
    indices_hit_list = []
    for hit_sequence in hit_sequences:
        idx = 0
        indices_hit = []
        for res in hit_sequence:
            if res == "-":
                indices_hit.append(-1)
            else:
                indices_hit.append(idx)
                idx += 1
        indices_hit_list.append(np.array(indices_hit, dtype=np.int32)[query_mask])
    return indices_hit_list


def _make_template_features_from_files(
    sequence: str,
    template_path: str,
    template_searcher: hhsearch.HHSearch,
    template_featurizer: templates.TemplateHitFeaturizer,
) -> Mapping[str, np.ndarray]:
    """Constructs a feature dict of template features.

    The template file is expected to be the same as AlphaFold template searching tools output.
    """

    with open(template_path, "r") as f:
        pdb_templates_result = f.read()

    pdb_template_hits = template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=sequence
    )
    templates_result = template_featurizer.get_templates(
        query_sequence=sequence, hits=pdb_template_hits
    )

    return templates_result.features


def _make_msa_features_for_all(
    msa_dir: str,
) -> pipeline.FeatureDict:
    """Constructs a feature dict of MSA features from a directory of MSA files."""
    logging.info("Making MSA features for all...")

    if not os.path.isdir(msa_dir):
        raise ValueError(f"MSA directory {msa_dir} not found.")

    msas = []
    for msa_file in os.listdir(msa_dir):
        msa_path = os.path.join(msa_dir, msa_file)
        if msa_file.endswith(".sto"):
            msa_result = run_msa_tool(
                msa_path=msa_path,
                msa_format="sto",
            )
            msa = parsers.parse_stockholm(msa_result["sto"])
        elif msa_file.endswith(".a3m"):
            msa_result = run_msa_tool(
                msa_path=msa_path,
                msa_format="a3m",
            )
            msa = parsers.parse_a3m(msa_result["a3m"])
        else:
            logging.info("File with unknown MSA format: %s, will be skipped", msa_file)
            continue
        msas.append(msa)

    if not msas:
        logging.info("No MSAs found in %s", msa_dir)
        return {}

    msa_features_for_all = pipeline.make_msa_features(msas)

    for key, value in msa_features_for_all.items():
        logging.info("%s: %s", key, value.shape)

    return msa_features_for_all


def _make_template_features_for_all(
    template_dir: str,
) -> GeneralFeatureDict:
    """Parses raw template features from a directory of PDB files."""
    logging.info("Making template features for all...")

    if not os.path.isdir(template_dir):
        raise ValueError(f"Template directory {template_dir} not found.")

    # Parse PDB files
    template_features_for_all: GeneralFeatureDict = {
        "all_atom_positions_list": [],
        "all_atom_masks_list": [],
        "sequence_list": [],
        "domain_name_list": [],
    }
    for pdb_name in os.listdir(template_dir):
        if not pdb_name.endswith(".pdb"):
            continue
        pdb_path = os.path.join(template_dir, pdb_name)
        all_atoms_positions, all_atoms_mask, sequence = parse_pdb(pdb_path)
        domain_name = os.path.basename(pdb_path).split(".")[0]

        template_features_for_all["all_atom_positions_list"].append(all_atoms_positions)
        template_features_for_all["all_atom_masks_list"].append(all_atoms_mask)
        template_features_for_all["sequence_list"].append(sequence)
        template_features_for_all["domain_name_list"].append(domain_name)

    if not template_features_for_all["sequence_list"]:
        logging.info("No templates found in %s", template_dir)
        return {}

    return template_features_for_all


def _make_sequence_features(
    sequence: str,
    domain_name: str,
) -> pipeline.FeatureDict:
    """Constructs a feature dict of sequence features."""
    logging.info("Making sequence features...")

    num_res = len(sequence)
    sequence_features: pipeline.FeatureDict = pipeline.make_sequence_features(
        sequence, domain_name, num_res
    )

    return sequence_features


def _make_msa_features(
    sequence: str,
    msa_features_for_all: pipeline.FeatureDict,
) -> pipeline.FeatureDict:
    """Constructs a feature dict of MSA features.

    Notes:
    ------
    - If msa_features_for_all is not provided, empty MSA features will be returned.
    - If msa_features_for_all is provided, the first MSA will be updated with the sequence.
    """

    logging.info("Making MSA features...")

    # Create empty MSA features
    num_res = len(sequence)
    empty_msa_features: pipeline.FeatureDict = {}
    empty_msa_features["deletion_matrix_int"] = np.zeros((1, num_res), dtype=np.int32)
    empty_msa_features["msa"] = np.array(
        [[residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]],
        dtype=np.int32,
    )
    empty_msa_features["num_alignments"] = np.ones(num_res, dtype=np.int32)
    empty_msa_features["msa_species_identifiers"] = np.array(
        ["".encode()], dtype=object
    )

    if not msa_features_for_all:
        return empty_msa_features
    if msa_features_for_all["msa"].shape[1] != len(sequence):
        raise ValueError("MSA and sequence lengths do not match.")

    msa_features = copy.deepcopy(msa_features_for_all)  # type: ignore
    msa_features["msa"][0] = [
        residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence
    ]

    return msa_features


def _make_template_features(
    sequence: str,
    template_features_for_all: GeneralFeatureDict,
    is_multimer: bool = False,
) -> pipeline.FeatureDict:
    """Constructs a feature dict of template features.

    Notes:
    ------
    - If template_features_for_all is not provided, empty template features will be returned.
    - If template_features_for_all is provided, the templates will be aligned to the sequence first,
        and the corresponding atom positions, atom masks and sequences features will be updated.
    """

    logging.info("Making template features...")

    # Create empty template features
    num_res = len(sequence)

    empty_template_features: pipeline.FeatureDict = (
        {
            "template_aatype": np.zeros(
                (1, num_res, len(residue_constants.restypes_with_x_and_gap)), np.float32
            ),
            "template_all_atom_masks": np.zeros(
                (1, num_res, residue_constants.atom_type_num), np.float32
            ),
            "template_all_atom_positions": np.zeros(
                (1, num_res, residue_constants.atom_type_num, 3), np.float32
            ),
            "template_domain_names": np.array(["".encode()], dtype=object),
            "template_sequence": np.array(["".encode()], dtype=object),
            "template_sum_probs": np.array([0], dtype=np.float32),
        }
        if is_multimer
        else {
            name: np.array([], dtype=dtype)
            for name, dtype in templates.TEMPLATE_FEATURES.items()
        }
    )

    # Check if self.template_features_for_all is empty
    if not template_features_for_all:
        return empty_template_features

    # Check if sequence is too short
    if num_res < 6:  # kalign has a minimum input length of 6
        logging.info("Template query is too short.")
        return empty_template_features

    # Unpack template features
    all_atoms_positions_list = template_features_for_all["all_atom_positions_list"]
    all_atoms_mask_list = template_features_for_all["all_atom_masks_list"]
    sequence_list = template_features_for_all["sequence_list"]
    domain_name_list = template_features_for_all["domain_name_list"]

    # Align sequences
    if (kalign_binary_path := shutil.which("kalign")) is not None:
        kalign_runner = kalign.Kalign(binary_path=kalign_binary_path)
    else:
        raise ValueError("kalign binary not found.")

    a3m = kalign_runner.align([sequence] + sequence_list)
    msa = parsers.parse_a3m(a3m)
    indices_hit_list = _msa_to_indices_hit(msa)

    template_features_raw: GeneralFeatureDict = {
        "template_aatype": [],
        "template_all_atom_positions": [],
        "template_all_atom_masks": [],
        "template_domain_names": [],
        "template_sequence": [],
        "template_sum_probs": [],
    }
    for (
        indices_hit,
        all_atoms_positions_,
        all_atoms_mask_,
        sequence_,
        domain_name,
    ) in zip(
        indices_hit_list,
        all_atoms_positions_list,
        all_atoms_mask_list,
        sequence_list,
        domain_name_list,
    ):
        # Index hit
        all_atoms_positions = np.where(
            indices_hit[:, None, None] == -1, 0, all_atoms_positions_[indices_hit]
        )
        all_atoms_mask = np.where(
            indices_hit[:, None] == -1, 0, all_atoms_mask_[indices_hit]
        )
        sequence = "".join(sequence_[idx] if idx != -1 else "-" for idx in indices_hit)
        logging.info("Template sequence %s: %s", domain_name, sequence)

        aatype = residue_constants.sequence_to_onehot(
            sequence, residue_constants.HHBLITS_AA_TO_ID
        )

        template_features_raw["template_aatype"].append(aatype)
        template_features_raw["template_all_atom_positions"].append(all_atoms_positions)
        template_features_raw["template_all_atom_masks"].append(all_atoms_mask)
        template_features_raw["template_domain_names"].append(domain_name.encode())
        template_features_raw["template_sequence"].append(sequence.encode())
        template_features_raw["template_sum_probs"].append([0.0])

    template_features: pipeline.FeatureDict = {
        key: np.array(value, dtype=templates.TEMPLATE_FEATURES[key])
        for key, value in template_features_raw.items()
    }

    return template_features


def _convert_pairing_msa_features(
    msa_features: pipeline.FeatureDict,
) -> pipeline.FeatureDict:
    """Converts MSA features to pairing MSA features."""
    valid_features = {
        "msa",
        "msa_mask",
        "deletion_matrix",
        "deletion_matrix_int",
        "msa_species_identifiers",
    }

    pairing_msa_features = {
        f"{key}_all_seq": value
        for key, value in msa_features.items()
        if key in valid_features
    }

    return pairing_msa_features


class DataPipeline:
    """Data pipeline for AlphaFold2 initial guess."""

    def __init__(
        self,
        msa_dir: str,
        template_dir: str,
    ) -> None:
        """Create an initial data pipeline object with features for all queries."""
        self.msa_dir = msa_dir
        self.template_dir = template_dir

        # Check if msa_dir is provided
        if msa_dir:
            self.msa_features_for_all = _make_msa_features_for_all(msa_dir)
        else:
            logging.info("No MSA provided.")
            self.msa_features_for_all = {}

        # Check if template_dir is provided
        if template_dir:
            self.template_features_for_all = _make_template_features_for_all(
                template_dir
            )
        else:
            logging.info("No templates provided.")
            self.template_features_for_all = {}

    def process(
        self,
        sequence: str,
        domain_name: str,
    ) -> pipeline.FeatureDict:
        """Constructs sequence, MSA and template features for each query sequence."""
        sequence_features = _make_sequence_features(sequence, domain_name)
        msa_features = _make_msa_features(sequence, self.msa_features_for_all)
        template_features = _make_template_features(
            sequence, self.template_features_for_all
        )

        feature_dict: pipeline.FeatureDict = {
            **sequence_features,
            **msa_features,
            **template_features,
        }

        return feature_dict


class DataPipelineSingleChain(DataPipeline):
    """Data pipeline for AlphaFold2 initial guess for a single chain in multimer prediction."""

    def __init__(
        self,
        msa_dir: str,
        template_dir: str,
        pairing_msa_dir: str,
    ) -> None:
        """Create a data pipeline object for a single chain in multimer prediction."""
        super().__init__(msa_dir, template_dir)

        self.pairing_msa_dir = pairing_msa_dir

        # Check if pairing_msa_dir is provided
        # only used for multimer mode
        if pairing_msa_dir:
            self.pairing_msa_features_for_all = _make_msa_features_for_all(
                pairing_msa_dir
            )
        else:
            logging.info("No pairing MSA provided.")
            self.pairing_msa_features_for_all = {}

    def process_single_chain(
        self,
        sequence: str,
        domain_name: str,
        is_homomer_or_monomer: bool,
    ) -> pipeline.FeatureDict:
        """Constructs sequence, MSA and template features for a single chain in query multimer sequence."""
        sequence_features = _make_sequence_features(sequence, domain_name)
        msa_features = _make_msa_features(sequence, self.msa_features_for_all)
        template_features = _make_template_features(
            sequence, self.template_features_for_all, is_multimer=True
        )

        feature_dict: pipeline.FeatureDict = {
            **sequence_features,
            **msa_features,
            **template_features,
        }

        if not is_homomer_or_monomer:
            pairing_msa_features = _convert_pairing_msa_features(
                _make_msa_features(sequence, self.pairing_msa_features_for_all)
            )
            feature_dict.update(pairing_msa_features)

        return feature_dict


class DataPipelineMultimer:
    """Data pipeline for AlphaFold2 initial guess for multimer prediction."""

    def __init__(
        self,
        precalc_dir: str,
        msa_dir_name: str = "",
        template_dir_name: str = "",
        pairing_msa_dir_name: str = "",
    ) -> None:
        """Create a data pipeline object for a multimer."""
        self.precalc_dir = precalc_dir
        self.msa_dir_name = msa_dir_name
        self.template_dir_name = template_dir_name
        self.pairing_msa_dir_name = pairing_msa_dir_name

        self.all_chain_features_for_all = self._make_all_chain_features_for_all()

    def _make_all_chain_features_for_all(self) -> dict[str, DataPipelineSingleChain]:
        """Initializes data pipelines for all chains in a multimer."""
        all_chain_features_for_all = {}
        for chain_id in protein.PDB_CHAIN_IDS:
            if not os.path.isdir(os.path.join(self.precalc_dir, chain_id)):
                continue

            # Get paths for MSA, template and pairing MSA directories
            dirs = {
                "msa_dir": os.path.join(self.precalc_dir, chain_id, self.msa_dir_name),
                "template_dir": os.path.join(
                    self.precalc_dir, chain_id, self.template_dir_name
                ),
                "pairing_msa_dir": os.path.join(
                    self.precalc_dir, chain_id, self.pairing_msa_dir_name
                ),
            }

            # Only use paths that exist, otherwise use empty string
            dirs = {
                key: value if os.path.isdir(value) else ""
                for key, value in dirs.items()
            }
            all_chain_features_for_all[chain_id] = DataPipelineSingleChain(**dirs)

        return all_chain_features_for_all

    def process(self, sequences: str, domain_name: str) -> pipeline.FeatureDict:
        """Constructs sequence, MSA and template features for each query multimer sequence."""

        seqs_list = sequences.split(":")
        if (seqs_num := len(seqs_list)) != (
            chains_num := len(self.all_chain_features_for_all)
        ):
            raise ValueError(
                f"Number of sequences ({seqs_num}) does not match number of chains ({chains_num})."
            )

        all_chain_features: dict[str, pipeline.FeatureDict] = {}
        sequence_features: dict[str, pipeline.FeatureDict] = {}
        is_homomer_or_monomer = len(set(seqs_list)) == 1
        for chain_id, sequence in zip(self.all_chain_features_for_all, seqs_list):
            logging.info("Processing chain %s", chain_id)
            if sequence in sequence_features:
                all_chain_features[chain_id] = copy.deepcopy(
                    sequence_features[sequence]
                )
                continue
            feature_dict = self.all_chain_features_for_all[
                chain_id
            ].process_single_chain(sequence, domain_name, is_homomer_or_monomer)
            chain_features = pipeline_multimer.convert_monomer_features(
                feature_dict, chain_id
            )
            all_chain_features[chain_id] = chain_features
            sequence_features[sequence] = chain_features

        all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)  # type: ignore
        np_example = feature_processing.pair_and_merge(all_chain_features)

        # Pad MSA to avoid zero-sized extra_msa
        feature_dict_multimer = pipeline_multimer.pad_msa(np_example, 512)

        # Add domain name for easy identification
        # will be deleted when running the model
        feature_dict_multimer["domain_name"] = np.array(
            [domain_name.encode()], dtype=object
        )

        return feature_dict_multimer


class Af2Ig(BaseModel):
    """AlphaFold2 initial guess model for monomer and multimer prediction."""

    def __init__(
        self,
        model_name: str,
        is_multimer: bool,
        data_dir: str,
        data_pipeline: DataPipeline | DataPipelineMultimer,
    ) -> None:
        """Initialize the AlphaFold2 model."""

        super().__init__()

        self.model_name = model_name
        self.is_multimer = is_multimer
        self.data_dir = data_dir
        self.data_pipeline = data_pipeline

        self.model = self._load_model()

    def _load_model(self) -> model.RunModel:
        """Loads the user-specified AlphaFold2 model."""
        model_config = config.model_config(self.model_name)
        if self.is_multimer:
            model_config.model.num_ensemble_eval = 1
        else:
            model_config.data.eval.num_ensemble = 1
        model_params = data.get_model_haiku_params(
            model_name=self.model_name, data_dir=self.data_dir
        )

        return model.RunModel(model_config, model_params)

    def predict(
        self,
        feature_dict: pipeline.FeatureDict,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        """Predicts structure using AlphaFold for the given sequence.

        Args:
        ----
        feature_dict: dict[str, np.ndarray]
            A dictionary of features generated by the data pipeline for the sequence.
        random_seed: int, optional
            The random seed for the model.

        Returns:
        -------
        dict[str, Any]
            The prediction results.
        """

        # Extract domain name and remove it from feature_dict
        domain_name = feature_dict.pop("domain_name").item().decode()
        logging.info("Predicting %s", domain_name)

        # Run the model
        processed_feature_dict = self.model.process_features(
            feature_dict, random_seed=random_seed
        )
        prediction_result = self.model.predict(
            processed_feature_dict, random_seed=random_seed
        )

        # Remove jax dependency from results
        prediction_result = _jnp_to_np(dict(prediction_result))

        # Add confidence metrics
        plddt = prediction_result["plddt"]
        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1
        )

        # Adjust chain id to start from 0
        # See https://github.com/google-deepmind/alphafold/issues/251
        if self.is_multimer:
            processed_feature_dict["asym_id"] -= 1  # type: ignore

        # Save the prediction as a PDB file
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not self.is_multimer,
        )
        unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)

        prediction_result["domain_name"] = domain_name
        prediction_result["unrelaxed_pdb"] = unrelaxed_pdb

        return prediction_result

    def save(
        self,
        prediction_result: dict[str, Any],
        output_dir: str,
    ) -> None:
        """Saves the model outputs to a directory.

        Args:
        ----
        prediction_result: dict[str, Any]
            The prediction results.
        output_dir: str
            The output directory to save the results.
        """

        # Extract domain name and unrelaxed PDB
        domain_name = prediction_result.pop("domain_name")
        unrelaxed_pdb = prediction_result.pop("unrelaxed_pdb")
        logging.info("Saving %s", domain_name)

        # Save the model outputs
        os.makedirs(output_dir, exist_ok=True)
        result_output_path = os.path.join(output_dir, f"result_{domain_name}.pkl")
        logging.info("Saving results to %s", result_output_path)
        with open(result_output_path, "wb") as f:
            pickle.dump(prediction_result, f, protocol=4)

        # Extract the pLDDT, PAE and pTM
        plddt = prediction_result["plddt"]
        average_plddt = np.mean(plddt)
        logging.info("Average pLDDT: %s", average_plddt)
        if "predicted_aligned_error" in prediction_result:
            pae = prediction_result["predicted_aligned_error"]
            average_pae = np.mean(pae)
            logging.info("Average PAE: %s", average_pae)
        if "ptm" in prediction_result:
            ptm = prediction_result["ptm"]
            logging.info("pTM score: %s", ptm)
        if "iptm" in prediction_result:
            iptm = prediction_result["iptm"]
            logging.info("ipTM score: %s", iptm)

        unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{domain_name}.pdb")
        logging.info("Saving PDB to %s", unrelaxed_pdb_path)
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdb)

        logging.info("Finished %s", domain_name)

    def _preprocess(self, q_in: queue.Queue, fasta_path: str) -> None:
        """Preprocesses the sequences and adds them to the queue."""
        with open(fasta_path, "r") as f:
            records = list(SimpleFastaParser(f))

        for domain_name, sequences in tqdm(records):
            logging.info("Sequence %s: %s", domain_name, sequences)

            # process features
            try:
                feature_dict = self.data_pipeline.process(sequences, domain_name)
                q_in.put(feature_dict)
            except ValueError as e:
                logging.error(e)

        q_in.put(None)

    def _process(
        self, q_in: queue.Queue, q_out: queue.Queue, random_seed: int = 0
    ) -> None:
        """Processes the sequences in the queue."""
        while True:
            feature_dict = q_in.get()
            if feature_dict is None:
                break
            prediction_result = self.predict(feature_dict, random_seed=random_seed)
            q_out.put(prediction_result)

        q_out.put(None)

    def _postprocess(self, q_out: queue.Queue, output_dir: str) -> None:
        """Saves the model outputs to a directory."""
        while True:
            prediction_result = q_out.get()
            if prediction_result is None:
                break
            self.save(prediction_result, output_dir)

    def _run(self, fasta_path: str, output_dir: str, random_seed: int = 0) -> None:
        """Runs AlphaFold2 initial guess for monomer and multimer prediction."""

        q_in: queue.Queue = queue.Queue(maxsize=3)
        q_out: queue.Queue = queue.Queue(maxsize=3)

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self._preprocess, q_in, fasta_path)
            executor.submit(self._process, q_in, q_out, random_seed=random_seed)
            executor.submit(self._postprocess, q_out, output_dir)

    def run(
        self,
        fasta_path: str,
        output_dir: str,
        random_seed: int = 0,
        asynchronous: bool = False,
    ) -> None:
        """Runs AlphaFold2 initial guess for monomer and multimer prediction.

        Args:
        ----
        fasta_path: str
            The path to the FASTA file containing the sequences.
        output_dir: str
            The output directory to save the results.
        random_seed: int, optional
            The random seed for the model.
        asynchronous: bool, optional
            Whether to run the pipeline asynchronously with multiple threads.

        Notes:
        ------
        - The asynchronous mode is experimental and may not cause a significant performance improvement.
        """

        if asynchronous:
            self._run(fasta_path, output_dir, random_seed=random_seed)
        else:
            with open(fasta_path, "r") as f:
                records = list(SimpleFastaParser(f))

            for domain_name, sequences in tqdm(records):
                logging.info("Sequence %s: %s", domain_name, sequences)

                # process features
                try:
                    feature_dict = self.data_pipeline.process(
                        sequences, domain_name
                    )
                    prediction_result = self.predict(
                        feature_dict, random_seed=random_seed
                    )
                    self.save(prediction_result, output_dir)
                except ValueError as e:
                    logging.error(e)


def cli(args: argparse.Namespace) -> None:
    """Runs AlphaFold2 initial guess for monomer and multimer prediction."""

    # Set logging level
    logging.set_verbosity(logging.INFO if args.verbose else logging.ERROR)

    # General arguments
    model_name = args.model_name
    data_dir = args.data_dir
    fasta_path = args.fasta_path
    output_dir = args.output_dir
    random_seed = args.random_seed

    is_multimer = "multimer" in model_name

    if is_multimer:
        # Multimer arguments
        precalc_dir = args.precalc_dir
        msa_dir_name = args.msa_dir_name
        template_dir_name = args.template_dir_name
        pairing_msa_dir_name = args.pairing_msa_dir_name
        data_pipeline = DataPipelineMultimer(
            precalc_dir, msa_dir_name, template_dir_name, pairing_msa_dir_name
        )
    else:
        # Monomer arguments
        msa_dir = args.msa_dir
        template_dir = args.template_dir
        data_pipeline = DataPipeline(msa_dir, template_dir)

    # Run AlphaFold2 initial guess
    af2ig = Af2Ig(model_name, is_multimer, data_dir, data_pipeline)

    with Timer() as timer:
        af2ig.run(
            fasta_path,
            output_dir,
            random_seed=random_seed,
            asynchronous=args.asynchronous,
        )

    logging.info("Total time: %s", timer.elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")

    # Monomer arguments
    parser.add_argument("--msa_dir", type=str, default="")
    parser.add_argument("--template_dir", type=str, default="")

    # Multimer arguments
    parser.add_argument("--precalc_dir", type=str, default="")
    parser.add_argument("--msa_dir_name", type=str, default="")
    parser.add_argument("--template_dir_name", type=str, default="")
    parser.add_argument("--pairing_msa_dir_name", type=str, default="")

    args = parser.parse_args()

    cli(args)


if __name__ == "__main__":
    main()
