import argparse
import os
import sys

import numpy as np
import torch

from models.data_utils import element_dict_rev, featurize, parse_PDB
from models.globals import AA_DICT
from models.model_utils import ProteinMPNN, cat_neighbors_nodes


class LigandMPNNBatch(ProteinMPNN):

    def __init__(
        self,
        num_letters: int = 21,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 32,
        augment_eps: float = 0.0,
        dropout: float = 0.0,
        device: str | torch.device | None = "cpu",
        atom_context_num: int = 25,
        model_type: str = "ligand_mpnn",
        ligand_mpnn_use_side_chain_context: bool = False,
    ) -> None:
        """Initialize the LigandMPNNBatch model.

        Args
        ------
        model_type: str
            The model to initialize.
            Has set to `ligand_mpnn` as default.
        k_neighbors: int
            The number of nearest residues neighbors to consider in the message passing.
            Has set to `32` as default.
        atom_context_num: int
            The number of nearest ligand atoms to consider in the message passing.
            Has set to `25` as default.
        """

        super().__init__(
            num_letters=num_letters,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout,
            device=device,
            atom_context_num=atom_context_num,
            model_type=model_type,
            ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
        )

    def single_aa_score(self, feature_dict, use_sequence: bool):
        """Rewrite the single_aa_score function of ProteinMPNN class to batch version.

        Allow different sequences to be scored in parallel on the same complex structure.
        """

        B_decoder = feature_dict["batch_size"]
        S_true_enc = feature_dict["S"]  # (B, L)
        mask_enc = feature_dict["mask"]
        chain_mask_enc = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        B, L = S_true_enc.shape  # B can be larger than 1
        device = S_true_enc.device

        h_V_enc, h_E_enc, E_idx_enc = self.encode(feature_dict)
        logits = torch.zeros([B_decoder, L, 21], device=device).float()

        for idx in range(L):
            h_V = torch.clone(h_V_enc)
            E_idx = torch.clone(E_idx_enc)
            mask = torch.clone(mask_enc)
            S_true = torch.clone(S_true_enc)

            if use_sequence:
                order_mask = torch.ones(chain_mask_enc.shape[1], device=device).float()
                order_mask[idx] = 0.0
            else:
                order_mask = torch.zeros(chain_mask_enc.shape[1], device=device).float()
                order_mask[idx] = 1.0
            decoding_order = torch.argsort(
                (order_mask + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )

            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([1, L, 1, 1])  # mask_1D[0] should be one
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            if S_true.shape[0] == 1:  # for compatibility with the source code
                S_true = S_true.repeat(B_decoder, 1)  # if duplicates are used
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E_enc.repeat(B_decoder, 1, 1, 1)
            mask = mask.repeat(B_decoder, 1)

            h_S = self.W_s(S_true)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits_idx = self.W_out(h_V)

            logits[:, idx, :] = logits_idx[:, idx, :]

        return {"logits": logits}


def score_complex(
    model: LigandMPNNBatch,
    pdbfile: str,
    seqs_list: list[str] | None = None,
    chains_to_design: str = "",
    redesigned_residues: str = "",
    use_side_chain_context: bool = False,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score protein sequences towards a given complex structure.

    Args
    ------
    model: LigandMPNNBatch
        The LigandMPNN model to use for scoring.
    pdbfile: str
        The path to the PDB file containing the complex structure.
    seqs_list: list[str] | None
        A list of sequences to score towards the complex structure.
        Chains are separated by colons.
    chains_to_design: str
        A comma-separated list of chain letters of the redigned chains.
        This option is only used to identify the regions not to use side chain context.
    redesigned_residues: str
        A space-separated list of chain letter and residue number pairs of the redesigned residues.
        This option is only used to identify the regions not to use side chain context.
    use_side_chain_context: bool
        Whether to use side chain context in the model.
        Use side chain context of all fixed residues if True, otherwise use only backbone context.
    verbose: bool
        Whether to print the parsed ligand atoms and their types.

    Returns
    -------
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
    - Both chains_to_design and redesigned_residues will have no effect if use_side_chain_context is False.
    - If seqs_list is empty or not provided, the native sequence is used as the target.
    - All sequences in the seqs_list should have the same length of each chain as the native sequence.
    """

    device = next(model.parameters()).device

    protein_dict = parse_PDB(
        pdbfile,
        device=device,
        parse_all_atoms=use_side_chain_context,
    )[0]

    chains_to_design_list = [
        chain.strip() for chain in chains_to_design.split(",") if chain.strip()
    ]  # chains_to_design.split(",") will not remove empty strings
    redesigned_residues_list = redesigned_residues.split()

    # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
    if chains_to_design_list or redesigned_residues_list:
        chain_letters = protein_dict["chain_letters"]
        R_idx = protein_dict["R_idx"].cpu().tolist()
        assert len(chain_letters) == len(R_idx)

        protein_dict["chain_mask"] = torch.tensor(
            [
                (chain_letter in chains_to_design_list)
                or (f"{chain_letter}{R_id}" in redesigned_residues_list)
                for chain_letter, R_id in zip(chain_letters, R_idx)
            ],
            device=device,
            dtype=torch.long,
        )
    else:
        protein_dict["chain_mask"] = torch.ones(
            protein_dict["R_idx"].shape[0], device=device, dtype=torch.long
        )

    # run featurize to remap R_idx and add batch dimension
    feature_dict = featurize(
        protein_dict,
        number_of_ligand_atoms=25,
        model_type="ligand_mpnn",
    )
    if seqs_list is None or not seqs_list:
        feature_dict["batch_size"] = (
            1  # modify to a larger integer if averaging over duplicates is desired
        )
    else:
        feature_dict["batch_size"] = len(seqs_list)
        feature_dict["S"] = torch.tensor(
            [[AA_DICT[aa] for aa in seqs.replace(":", "")] for seqs in seqs_list],
            device=device,
        )

    # sample random decoding order
    feature_dict["randn"] = torch.randn(
        feature_dict[
            "batch_size"
        ],  # batch_size may not equal size of target_seqs_list when duplicates are used
        feature_dict["S"].shape[1],
        device=device,
    )

    use_sequence = True
    with torch.no_grad():
        logits = model.single_aa_score(feature_dict, use_sequence)["logits"]
    logits = logits.cpu().detach()

    entropy = -(logits[:, :, :20].log_softmax(dim=-1))  # (B, L, 20)
    target = feature_dict["S"].long().cpu()  # (B, L)
    loss = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)
    perplexity = torch.exp(loss.mean(dim=-1))  # (B,)

    if verbose:
        atom_masks = protein_dict.get("Y_m", torch.tensor([])).cpu().numpy()
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

    return entropy, loss, perplexity


def extract_from_score(output_path):
    """Extract entropy, loss, and perplexity from LigandMPNN output score file."""
    with open(output_path, "rb") as f:
        output = torch.load(f)

    entropy = -(
        torch.tensor(output["logits"][:, :, :20]).softmax(dim=-1).mean(dim=0).log()
    )  # (L, 20)
    target = torch.tensor(output["native_sequence"], dtype=torch.long)  # (L,)
    loss = torch.gather(entropy, 1, target.unsqueeze(1)).squeeze()  # (L,)
    perplexity = torch.exp(loss.mean()).item()  # scalar

    return entropy, loss, perplexity


def extract_from_sample(output_path):
    """Extract entropy, loss, and perplexity from LigandMPNN output sample file."""
    with open(output_path, "rb") as f:
        output = torch.load(f)

    entropy = -output["log_probs"]  # (B, L, 20)
    target = output["generated_sequences"]  # (B, L)
    loss = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)
    perplexity = torch.exp(loss.mean(dim=-1))  # (B,)
    # redesigned = output["chain_mask"] == 1
    # confidence = torch.exp(-loss[:, redesigned].mean(dim=-1))

    return entropy, loss, perplexity


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        "../model_params/ligandmpnn_v_32_020_25.pt",
        map_location=device,
        weights_only=True,
    )
    ligand_mpnn = LigandMPNNBatch(
        model_type="ligand_mpnn",
        k_neighbors=32,
        atom_context_num=25,
        ligand_mpnn_use_side_chain_context=args.use_side_chain_context,
        device=device,
    )
    ligand_mpnn.load_state_dict(checkpoint["model_state_dict"])
    ligand_mpnn.to(device)
    ligand_mpnn.eval()

    entropy, loss, perplexity = score_complex(
        ligand_mpnn,
        args.pdbfile,
        seqs_list=args.seqs_list,
        chains_to_design=args.chains_to_design,
        redesigned_residues=args.redesigned_residues,
        use_side_chain_context=args.use_side_chain_context,
        verbose=args.verbose,
    )

    out_dict = {
        "entropy": entropy,
        "loss": loss,
        "perplexity": perplexity,
    }

    torch.save(out_dict, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbfile", type=str, help="Path to the PDB file.")
    parser.add_argument("output_path", type=str, help="Path to save the output.")
    parser.add_argument(
        "--seqs_list",
        type=str,
        nargs="+",
        default=None,
        help="A list of sequences to score towards the complex structure.",
    )
    parser.add_argument(
        "--chains_to_design", type=str, default="", help="Chains to design."
    )
    parser.add_argument(
        "--redesigned_residues", type=str, default="", help="Redesigned residues."
    )
    parser.add_argument(
        "--use_side_chain_context", action="store_true", help="Use side chain context."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print parsed ligand atoms and types."
    )
    args = parser.parse_args()

    main(args)
