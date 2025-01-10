import random
from typing import Any, Literal, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from models.basemodels import TorchModel
from models.globals import AA_DICT, ScoreDict
from models.ligandmpnn.data_utils import element_dict_rev, featurize, parse_PDB
from models.ligandmpnn.model_utils import ProteinMPNN, cat_neighbors_nodes


class FeatureDict(TypedDict):
    """Type hint for the feature dictionary used in the MPNN models."""

    S: torch.Tensor
    mask: torch.Tensor
    chain_mask: torch.Tensor
    randn: torch.Tensor
    symmetry_residues: list[list[int]]
    batch_size: int


class ProteinMPNNBatch(ProteinMPNN):
    """The ProteinMPNN model class for batch sampling and scoring."""

    # TODO: Modify the sample method

    @staticmethod
    def _symmetric_decoding_order(
        decoding_order: torch.Tensor,
        symmetry_list_of_lists: list[list[int]],
    ) -> torch.Tensor:
        """Aggregate symmetry residues together and update decoding order."""
        device = decoding_order.device

        new_decoding_orders = []
        for order in decoding_order.cpu().numpy():
            new_decoding_order: list[int] = []
            for t in order:
                if t in new_decoding_order:
                    continue
                for symmetry_list in symmetry_list_of_lists:
                    if t in symmetry_list:
                        new_decoding_order.extend(symmetry_list)
                        break
                else:
                    new_decoding_order.append(t)
            new_decoding_orders.append(torch.tensor(new_decoding_order, device=device))
        decoding_order = torch.stack(new_decoding_orders)

        return decoding_order

    def _process_features(self, feature_dict: FeatureDict) -> dict[str, torch.Tensor]:
        """Prepare tensors for the encoder and decoder."""
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"]  # (B, L)
        mask = feature_dict["mask"]

        B, L = S_true.shape  # B can be larger than 1
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        E_idx = E_idx.repeat(B_decoder, 1, 1)

        if B == 1:  # for compatibility with the source code
            S_true = S_true.repeat(B_decoder, 1)  # if duplicates are used
        mask_1D = mask.view([1, L, 1, 1])  # mask_1D[0] should be one
        mask = mask.repeat(B_decoder, 1)

        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros(B_decoder, L, self.hidden_dim, device=device), h_E, E_idx
        )
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        h_S = self.W_s(S_true)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        processed_features = {
            "h_V": h_V,
            "E_idx": E_idx,
            "mask": mask,
            "mask_1D": mask_1D,
            "h_EXV_encoder": h_EXV_encoder,
            "h_ES": h_ES,
        }

        return processed_features

    def decode(self, processed_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """The decoding method for the ProteinMPNNBatch class."""
        h_V = processed_features["h_V"]
        E_idx = processed_features["E_idx"]
        mask = processed_features["mask"]
        mask_1D = processed_features["mask_1D"]
        h_EXV_encoder = processed_features["h_EXV_encoder"]
        h_ES = processed_features["h_ES"]
        decoding_order = processed_features["decoding_order"]

        L = mask.shape[1]
        device = mask.device

        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(L, L, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)

        return logits

    def forward(self, feature_dict: FeatureDict) -> dict[str, torch.Tensor]:
        """The forward method for ProteinMPNNBatch class.

        Used as the score method in the ProteinMPNN class.
        """

        processed_features = self._process_features(feature_dict)

        mask = processed_features["mask"]
        chain_mask = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        symmetry_list_of_lists = feature_dict["symmetry_residues"]
        decoding_order = torch.argsort(
            (chain_mask * mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        # aggregate symmetry residues together and update decoding order
        if symmetry_list_of_lists:
            decoding_order = self._symmetric_decoding_order(
                decoding_order, symmetry_list_of_lists
            )
        processed_features["decoding_order"] = decoding_order

        logits = self.decode(processed_features)

        return {
            "logits": logits,
        }

    def single_aa_score(
        self, feature_dict: FeatureDict, use_sequence: bool
    ) -> dict[str, torch.Tensor]:
        """Rewrite the score and single_aa_score method of ProteinMPNN class to batch version.

        Allow different sequences to be scored in parallel on the same complex structure.
        """

        processed_features = self._process_features(feature_dict)

        mask = processed_features["mask"]

        if use_sequence:
            randn = feature_dict["randn"]
            symmetry_list_of_lists = feature_dict["symmetry_residues"]

            L = mask.shape[1]
            device = mask.device

            logits_list = []
            for idx in range(L):
                # mask the target residue and symmetric residues
                order_mask = torch.zeros(L, device=device)
                for symmetry_list in symmetry_list_of_lists:
                    if idx in symmetry_list:
                        order_mask[symmetry_list] = 1
                        break
                else:
                    order_mask[idx] = 1
                decoding_order = torch.argsort(
                    (order_mask + 0.0001) * (torch.abs(randn))
                )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
                # aggregate symmetry residues together
                if symmetry_list_of_lists:
                    decoding_order = self._symmetric_decoding_order(
                        decoding_order, symmetry_list_of_lists
                    )
                processed_features["decoding_order"] = decoding_order

                logits_idx = self.decode(processed_features)
                logits_list.append(logits_idx[:, idx])

            logits = torch.stack(logits_list, dim=1)
        else:
            h_V = processed_features["h_V"]
            mask_1D = processed_features["mask_1D"]
            h_EXV_encoder = processed_features["h_EXV_encoder"]

            h_EXV_encoder_fw = mask_1D * h_EXV_encoder
            for layer in self.decoder_layers:
                h_V = layer(h_V, h_EXV_encoder_fw, mask)

            logits = self.W_out(h_V)

        return {
            "logits": logits,
        }


class MPNN(TorchModel):
    """The MPNN model interface for batch sampling and scoring."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: Literal["protein_mpnn", "ligand_mpnn"] = "protein_mpnn",
        device: str | torch.device | None = None,
        ligand_mpnn_use_side_chain_context: bool = False,
        ligand_mpnn_use_atom_context: bool = True,
        ligand_mpnn_cutoff_for_score: float = 8.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the MPNN model from the checkpoint file.

        Args
        ------
        checkpoint_path: str
            The path to the checkpoint file.
        model_type: str
            The type of the MPNN model.
        device: str or torch.device
            The device to run the model on.
        ligand_mpnn_use_side_chain_context: bool
            Whether to use side chain context for the LigandMPNN model.
        ligand_mpnn_use_atom_context: bool
            Whether to use atom context for the LigandMPNN model.
        ligand_mpnn_cutoff_for_score: float
            The cutoff distance for the LigandMPNN model.
        seed: int
            The seed to set for reproducibility.

        Notes
        ------
        - If use_side_chain_context is False, both chains_to_design and redesigned_residues
            in the score method will have no effect
        """

        super().__init__(device=device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.checkpoint = checkpoint

        if model_type == "ligand_mpnn":
            atom_context_num = checkpoint["atom_context_num"]
        else:
            atom_context_num = 1
            ligand_mpnn_use_side_chain_context = False
            ligand_mpnn_use_atom_context = False
            ligand_mpnn_cutoff_for_score = 0
        k_neighbors = checkpoint["num_edges"]

        self.model_type = model_type

        self.atom_context_num = atom_context_num
        self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
        self.ligand_mpnn_use_atom_context = ligand_mpnn_use_atom_context
        self.ligand_mpnn_cutoff_for_score = ligand_mpnn_cutoff_for_score
        self.k_neighbors = k_neighbors

        self.seed = seed
        self.model = self._load_model()

    def _load_model(self) -> ProteinMPNNBatch:
        """Load the MPNN model from the checkpoint file."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

        model = ProteinMPNNBatch(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=self.k_neighbors,
            device=self.device,
            atom_context_num=self.atom_context_num,
            model_type=self.model_type,
            ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context,
        )

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def _parse_PDB(
        self,
        pdb_path: str,
    ) -> dict[str, Any]:
        """Parse the PDB file and create the feature dictionary for the MPNN model."""
        protein_dict = parse_PDB(
            pdb_path,
            device=self.device,
            parse_all_atoms=self.ligand_mpnn_use_side_chain_context,
        )[0]

        return protein_dict

    def _get_chain_mask(
        self,
        chain_letters: list[str],
        R_idx: list[int],
        chains_to_design: str,
        redesigned_residues: str,
        use_sequence: bool = True,
    ) -> torch.Tensor:
        """Create a chain mask to notify which residues are fixed (0) and which need to be designed (1).

        Args
        ------
        chain_letters: list[str]
            A list of chain letters of the protein structure.
        R_idx: list[int]
            A list of residue indices of the protein structure.
        chains_to_design: str
            A comma-separated list of chain letters of the redesign chains.
            This option is only used to identify the regions not to use side chain context.
        redesigned_residues: str
            A space-separated list of chain letter and residue number pairs of the redesigned residues.
            This option is only used to identify the regions not to use side chain context.
        use_sequence: bool
            Whether to use the sequence information in the scoring method.

        Returns
        -------
        chain_mask: torch.Tensor
            A binary tensor notifying which residues are redesignable.
        """

        if not use_sequence:
            return torch.ones(len(R_idx), device=self.device)

        chains_to_design_list = [
            chain.strip() for chain in chains_to_design.split(",") if chain.strip()
        ]  # chains_to_design.split(",") will not remove empty strings
        redesigned_residues_list = redesigned_residues.split()

        if chains_to_design_list or redesigned_residues_list:
            chain_mask = [
                (chain_letter in chains_to_design_list)
                or (f"{chain_letter}{R_id}" in redesigned_residues_list)
                for chain_letter, R_id in zip(chain_letters, R_idx)
            ]
            return torch.tensor(
                chain_mask,
                device=self.device,
            ).float()
        else:
            return torch.ones(len(R_idx), device=self.device)

    def _featurize(
        self,
        protein_dict: dict[str, Any],
    ) -> FeatureDict:
        """Featurize the protein structure and sequences for the MPNN model."""

        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=self.ligand_mpnn_cutoff_for_score,
            use_atom_context=self.ligand_mpnn_use_atom_context,
            number_of_ligand_atoms=self.atom_context_num,
            model_type=self.model_type,
        )

        return feature_dict

    def _parse_symmetry_residues(
        self, symmetry_residues: str, res_to_idx: dict[str, int]
    ) -> list[list[int]]:
        """Parse the symmetry residues and group them together.

        Make sure that the symmetry residues are non-empty and pairwise disjoint.

        Args
        ------
        symmetry_residues: str
            A pipe-separated list of comma-separated symmetry residues.
        res_to_idx: dict[str, int]
            A dictionary mapping residue names to their indices.

        Returns
        -------
        symmetry_list_of_lists: list[list[int]]
            A list of lists of symmetry residues grouped together.
        """

        symmetry_list_of_lists: list[list[int]] = []
        for symmetry_group in symmetry_residues.split("|"):
            symmetry_set: set[int] = {
                res_to_idx[res.strip()]
                for res in symmetry_group.split(",")
                if res.strip() in res_to_idx
            }
            if not symmetry_set:
                continue
            symmetry_list_of_lists_copy: list[list[int]] = []
            for symmetry_list in symmetry_list_of_lists:
                if symmetry_set.intersection(symmetry_list):
                    symmetry_set.update(symmetry_list)
                else:
                    symmetry_list_of_lists_copy.append(symmetry_list)
            symmetry_list_of_lists = symmetry_list_of_lists_copy
            symmetry_list_of_lists.append(list(symmetry_set))

        return symmetry_list_of_lists

    def _print_verbose_message(self, protein_dict: dict[str, Any]) -> None:
        """Print the parsed ligand atoms and their types."""
        atom_masks = protein_dict.get("Y_m", torch.tensor([0])).cpu().numpy()
        if lig_atom_num := np.sum(atom_masks):
            atom_coords = protein_dict["Y"].cpu().numpy()
            atom_types = protein_dict["Y_t"].cpu().numpy()
            print(f"The number of ligand atoms parsed is equal to: {lig_atom_num}")
            for atom_type, atom_coord, atom_mask in zip(
                atom_types, atom_coords, atom_masks
            ):
                print(
                    f"Type: {element_dict_rev[atom_type]}, Coords {atom_coord}, Mask {atom_mask}"
                )
        else:
            print("No ligand atoms parsed")

    def score(
        self,
        pdb_path: str,
        seqs_list: Sequence[str] | None = None,
        chains_to_design: str = "",
        redesigned_residues: str = "",
        symmetry_residues: str = "",
        autoregressive_score: bool = False,
        use_sequence: bool = True,
        verbose: bool = False,
    ) -> ScoreDict:
        """Score protein sequences towards a given complex structure.

        Args
        ------
        pdb_path: str
            The path to the PDB file containing the complex structure.
        seqs_list: Sequence[str]
            A list of sequences to score towards the complex structure.
            Chains are separated by colons.
        chains_to_design: str
            A comma-separated list of chain letters of the redesign chains.
        redesigned_residues: str
            A space-separated list of chain letter and residue number pairs of the redesigned residues.
        symmetry_residues: str
            A pipe-separated list of comma-separated symmetry residues.
        autoregressive_score: bool
            Whether to use the autoregressive scoring method.
        use_sequence: bool
            Whether to use the sequence information in the scoring method.
        verbose: bool
            Whether to print the parsed ligand atoms and their types.

        Returns
        -------
        output_dict: ScoreDict
            entropy: torch.Tensor[B, L, 20]
                -log{logits} of the masked token at each position.
            loss: torch.Tensor[B, L]
                Cross entropy of the true residue at each position.
            perplexity: torch.Tensor[B,]
                exp{average entropy} of the full sequence.

        Notes
        ------
        - If chains_to_design and redesigned_residues are empty, all residues are considered redesignable.
        - If chains_to_design and redesigned_residues are both provided, the union of the two is used.
        - The chains_to_design and redesigned_residues flags are only used
            to identify the regions not to use side chain context in the structural encoder.
        - If seqs_list is empty or not provided, the native sequence is used as the target.
        - All sequences in the seqs_list should have the same length of each chain as the native sequence.
        - The use_sequence option is only used to determine if sequence information is used in the sequence decoder.
        """

        protein_dict = self._parse_PDB(pdb_path)

        # create chain mask
        chain_letters = protein_dict["chain_letters"]
        R_idx = protein_dict["R_idx"].cpu().tolist()
        chain_mask = self._get_chain_mask(
            chain_letters, R_idx, chains_to_design, redesigned_residues
        )
        protein_dict["chain_mask"] = chain_mask

        feature_dict = self._featurize(protein_dict)

        # remap R_idx and add batch dimension
        if seqs_list is None or not seqs_list:
            feature_dict["batch_size"] = (
                1  # modify to a larger integer if averaging over duplicates is desired
            )
        else:
            feature_dict["batch_size"] = len(seqs_list)
            feature_dict["S"] = torch.tensor(
                [[AA_DICT[aa] for aa in seqs.replace(":", "")] for seqs in seqs_list],
                device=self.device,
            )

        # sample random decoding order
        feature_dict["randn"] = torch.randn(
            feature_dict[
                "batch_size"
            ],  # batch_size may not equal size of target_seqs_list when duplicates are used
            feature_dict["S"].shape[1],
            device=self.device,
        )

        # remap symmetry residues
        res_to_idx = {
            f"{chain_letter}{R_id}": idx
            for idx, (chain_letter, R_id) in enumerate(zip(chain_letters, R_idx))
        }
        symmetry_list_of_lists = self._parse_symmetry_residues(
            symmetry_residues, res_to_idx
        )
        feature_dict["symmetry_residues"] = symmetry_list_of_lists

        with torch.no_grad():
            if autoregressive_score and use_sequence:
                score_dict = self.model(feature_dict)
            else:
                score_dict = self.model.single_aa_score(feature_dict, use_sequence)
            logits = score_dict["logits"]
        logits = logits.cpu().detach()

        entropy = -(logits[:, :, :20].log_softmax(dim=-1))  # (B, L, 20)
        target = feature_dict["S"].long().cpu()  # (B, L)
        loss = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)
        perplexity = torch.exp(loss.mean(dim=-1))  # (B,)

        if verbose:
            self._print_verbose_message(protein_dict)

        output_dict = {
            "entropy": entropy,
            "loss": loss,
            "perplexity": perplexity,
        }

        return output_dict
