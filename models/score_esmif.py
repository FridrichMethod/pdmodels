import argparse

import esm
import numpy as np
import torch
import torch_geometric
import torch_sparse
from esm.data import Alphabet
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.multichain_util import (
    _concatenate_coords,
    extract_coords_from_complex,
)
from esm.inverse_folding.util import CoordBatchConverter, load_structure
from torch_geometric.nn import MessagePassing

from models.globals import AA_ALPHABET, AA_DICT, CHAIN_ALPHABET


def _concatenate_seqs(
    target_seqs: dict[str, str],
    target_chain_id: str,
    padding_length: int = 10,
) -> tuple[str, np.ndarray]:
    """Concatenate the chain sequences with padding in between.

    Args:
        target_seqs (dict[str, str]):
            Modified target dictionary mapping chain ids to corresponding AA sequence
        target_chain_id (str):
            The chain id to sample sequences for
        padding_length (int):
            Length of padding between concatenated chains

    Returns:
        (target_seqs_concatenated, target_aa_indices) (tuple):
            - target_seqs_concatenated (str): Concatenated sequences with padding in between
            - target_aa_indices (np.ndarray): Indices of the concatenated sequences in the concatenated array
    """

    target_seq = target_seqs[target_chain_id]
    target_seqs_list = [target_seq]
    target_aa_indices = []

    i = len(target_seq)
    for chain_id, seq in target_seqs.items():
        if chain_id == target_chain_id:
            target_aa_indices.append(np.arange(len(target_seq)))
        else:
            i += padding_length
            target_seqs_list.append(seq)
            target_aa_indices.append(np.arange(i, i + len(seq)))
            i += len(seq)

    target_seqs_concatenated = ("<mask>" * (padding_length - 1) + "<cath>").join(
        target_seqs_list
    )
    target_aa_concatenated = np.concatenate(target_aa_indices, axis=0)

    return target_seqs_concatenated, target_aa_concatenated


def score_complex(
    model: GVPTransformerModel,
    alphabet: Alphabet,
    pdbfile: str,
    target_seq_list: list[str] | None = None,
    target_chain_id: str = "A",
    padding_length: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score target sequences towards a given complex structure.

    Args
    ----
    model: GVPTransformerModel
        The GVPTransformerModel model from ESM-IF.
    alphabet: Alphabet
        The alphabet used for encoding the sequences.
    pdbfile: str
        The path to the PDB file.
    target_seq_list: list[str] | None
        A list of sequences of the same single chain to score towards the complex structure.
    target_chain_id: str
        The chain id of the target sequence.
    padding_length: int
        Padding length for chain separation.
    verbose: bool
        Whether to print the results.

    Returns
    -------
    entropy: torch.Tensor[B, L, 20]
        -log{logits} of the masked token at each position.
    loss: torch.Tensor[B, L]
        Cross entropy of the true residue at each position.
    perplexity: torch.Tensor[B,]
        exp{average entropy} of the full sequence.

    Notes:
    ------
    - The target sequences should be of the same length as the target chain in the native complex.
    """

    device = next(model.parameters()).device

    struct = load_structure(pdbfile)
    native_coords, native_seqs = extract_coords_from_complex(struct)
    if target_seq_list is None:
        target_seq_list = [native_seqs[target_chain_id]]
    assert all(
        len(target_seq) == len(native_seqs[target_chain_id])
        for target_seq in target_seq_list
    )
    target_seqs_list = [
        {
            chain_id: (target_seq if chain_id == target_chain_id else seq)
            for chain_id, seq in native_seqs.items()
        }
        for target_seq in target_seq_list
    ]
    aa_tokens = torch.tensor(alphabet.encode(AA_ALPHABET))

    all_coords = _concatenate_coords(
        native_coords,
        target_chain_id,
        padding_length=padding_length,
    )

    all_seqs_list: list[str] = []
    all_indices_list: list[np.ndarray] = []
    for target_seqs in target_seqs_list:
        all_seqs, all_indices = _concatenate_seqs(
            target_seqs,
            target_chain_id,
            padding_length=padding_length,
        )
        all_seqs_list.append(all_seqs)
        all_indices_list.append(all_indices)
    all_indices_array = np.array(all_indices_list)
    assert np.all(np.diff(all_indices_array, axis=0) == 0)

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(all_coords, None, all_seqs) for all_seqs in all_seqs_list]
    coords, confidence, _, tokens, padding_mask = batch_converter(batch, device=device)
    prev_output_tokens = tokens[:, :-1]  # shift tokens to predict next token's logits
    B, L = prev_output_tokens.shape

    with torch.no_grad():
        logits, _ = model(coords, padding_mask, confidence, prev_output_tokens)
    logits = logits.cpu().detach()
    entropy = -(
        logits[:, aa_tokens]
        .transpose(1, 2)[np.arange(B)[:, None], all_indices_array]
        .log_softmax(dim=-1)
    )  # (B, L, 20)

    target = torch.tensor(
        [
            [AA_DICT[aa] for aa in "".join(target_seqs.values())]
            for target_seqs in target_seqs_list
        ]
    )  # (B, L)
    loss = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)
    perplexity = torch.exp(loss.mean(dim=-1))  # (B,)

    if verbose:
        print(f"native_seqs: {native_seqs}")
        for l, target_seqs in enumerate(target_seqs_list):
            print()
            print(f"target_seq: {target_seqs[target_chain_id]}")
            print("loss:")
            k = 0
            for i, chain in enumerate(target_seqs.values()):
                loss_chunk = loss[l, k : k + len(chain)]
                k += len(chain)
                print(f"chain {CHAIN_ALPHABET[i]}")
                for j, (aa, loss_val) in enumerate(zip(chain, loss_chunk)):
                    print(f"{aa}{j+1}: {loss_val.item()}")
            print(f"perplexity: {perplexity[l].item()}")
            print()

    return entropy, loss, perplexity


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esmif_model, esmif_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esmif_model = esmif_model.to(device)
    esmif_model = esmif_model.eval()

    entropy, loss, perplexity = score_complex(
        esmif_model,
        esmif_alphabet,
        args.pdbfile,
        target_seq_list=args.target_seq_list,
        target_chain_id=args.target_chain_id,
        padding_length=args.padding_length,
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
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output.",
    )
    parser.add_argument(
        "--target_seq_list",
        type=str,
        nargs="+",
        default=None,
        help="A list of sequences of the same single chain to score towards the complex structure.",
    )
    parser.add_argument(
        "--target_chain_id",
        type=str,
        default="A",
        help="The chain id of the target sequence.",
    )
    parser.add_argument(
        "--padding_length",
        type=int,
        default=10,
        help="Padding length for chain separation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the loss and perplexity of the complex.",
    )
    args = parser.parse_args()

    main(args)
