import argparse
import os
import sys

import esm
import numpy as np
import torch
from esm.data import Alphabet
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.multichain_util import (
    _concatenate_coords,
    extract_coords_from_complex,
)
from esm.inverse_folding.util import CoordBatchConverter, load_structure

from models.globals import AA_ALPHABET, AA_DICT, CHAIN_ALPHABET


def _concatenate_seqs(
    target_seqs: dict[str, str],
    target_chain_id: str,
    padding_length: int = 10,
) -> tuple[str, np.ndarray]:
    """Concatenate the chain sequences with padding in between.

    Args:
    ----
    target_seqs: dict[str, str]:
        Modified target dictionary mapping chain ids to corresponding AA sequence
    target_chain_id: str
        The chain id to sample sequences for
    padding_length: int
        Length of padding between concatenated chains

    Returns:
    -------
    target_seqs_concatenated: str
        Concatenated sequences with padding in between
    target_aa_indices: np.ndarray
        Indices of the concatenated sequences in the concatenated array
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


class ESMIF:
    """ESM-IF model for scoring and sampling redesigned sequences for a given complex structure."""

    def __init__(self, device: str | torch.device | None = None):
        self.device = device
        self.model, self.alphabet = self._load_model()

    def _load_model(self) -> tuple[GVPTransformerModel, Alphabet]:
        """Load the ESM-IF model and the corresponding alphabet."""
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.to(self.device)
        model = model.eval()

        return model, alphabet  # type: ignore

    def to(self, device: str | torch.device | None) -> None:
        """Move the model to the given device."""
        self.device = device
        self.model = self.model.to(self.device)  # type: ignore

    def score(
        self,
        pdb_path: str,
        target_seq_list: list[str] | None = None,
        target_chain_id: str = "A",
        padding_length: int = 10,
        verbose: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Score target sequences towards a given complex structure.

        Args
        ----
        pdb_path: str
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
        output_dict: dict[str, torch.Tensor]
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

        struct = load_structure(pdb_path)
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
        aa_tokens = torch.tensor(self.alphabet.encode(AA_ALPHABET))

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

        batch_converter = CoordBatchConverter(self.alphabet)
        batch = [(all_coords, None, all_seqs) for all_seqs in all_seqs_list]
        coords, confidence, _, tokens, padding_mask = batch_converter(
            batch, device=self.device
        )

        # shift tokens to predict next token's logits
        prev_output_tokens = tokens[:, :-1]
        B, L = prev_output_tokens.shape

        with torch.no_grad():
            logits, _ = self.model(coords, padding_mask, confidence, prev_output_tokens)
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

        output_dict = {
            "entropy": entropy,
            "loss": loss,
            "perplexity": perplexity,
        }

        return output_dict

    def sample(
        self,
        pdb_path: str,
        output_path: str,
        target_chain_id: str = "A",
        batch_size: int = 1,
        redesigned_residues: str = "",
        omit_aa: str = "",
        temperature: float = 1.0,
        padding_length: int = 10,
        index_offset: int = 0,
    ) -> None:
        """Sample redesigned sequences for a given complex structure.

        Args
        ----
        pdb_path: str
            Path to the PDB file.
        output_path: str
            Path to save the sampled sequences.
        target_chain_id: str
            Chain ID of the target sequence.
        batch_size: int
            Number of sequences to sample.
        redesigned_residues: str
            Space-separated indices of redesigned residues.
        omit_aa: str
            Amino acids to omit from the sampling.
        temperature: float
            Sampling temperature.
        padding_length: int
            Length of padding between concatenated chains.
        index_offset: int
            Offset for the sequence IDs.
        """

        struct = load_structure(pdb_path)
        native_coords, native_seqs = extract_coords_from_complex(struct)
        all_coords = _concatenate_coords(
            native_coords, target_chain_id, padding_length=padding_length
        )

        mask_idx = self.alphabet.get_idx("<mask>")
        cath_idx = self.alphabet.get_idx("<cath>")
        pad_idx = self.alphabet.get_idx("<pad>")

        # Mask out redesigned residues of the target sequence
        target_seq_tokens = torch.tensor(
            self.alphabet.encode(native_seqs[target_chain_id])
        )
        redesigned_residues_positions = torch.tensor(
            [int(i) - 1 for i in redesigned_residues.split()]
        )
        target_seq_tokens[redesigned_residues_positions] = mask_idx

        B = batch_size
        L = all_coords.shape[0]  # length of the whole complex sequence with padding
        l = target_seq_tokens.shape[0]  # length of the target sequence

        # Start with prepend token, followed by the target sequence with masked redesigned residues, and padding to the end
        sampled_tokens = torch.full((B, L + 1), pad_idx, dtype=torch.long)  # (B, L + 1)
        sampled_tokens[:, 0] = cath_idx
        sampled_tokens[:, 1 : l + 1] = target_seq_tokens
        sampled_tokens = sampled_tokens.to(self.device)

        # Tokens to remove from the output distribution
        removed_aa_tokens = torch.tensor(
            self.alphabet.encode(
                "".join(
                    aa
                    for aa in self.alphabet.all_toks
                    if (aa not in AA_ALPHABET) or (aa in omit_aa)
                )
            )
        )

        batch_converter = CoordBatchConverter(self.alphabet)
        batch = [(all_coords, None, None) for _ in range(B)]
        batch_coords, confidence, _, _, padding_mask = batch_converter(
            batch, device=self.device
        )

        # Save incremental states for faster sampling
        incremental_state = {}

        with torch.no_grad():
            # Run encoder only once
            encoder_out = self.model.encoder(batch_coords, padding_mask, confidence)

            # Autoregressively decode the sequence one token at a time
            for j in range(1, l + 1):
                logits, _ = self.model.decoder(
                    sampled_tokens[:, :j],
                    encoder_out,
                    incremental_state=incremental_state,  # incremental decoding
                )
                if sampled_tokens[0, j] == mask_idx:
                    assert torch.all(sampled_tokens[:, j] == mask_idx)

                    logits[:, removed_aa_tokens] = -torch.inf
                    logits = logits.transpose(1, 2) / temperature  # (B, 1, V)
                    probs = logits.softmax(dim=-1)[:, -1]  # (B, V)

                    sampled_tokens[:, j] = torch.multinomial(probs, 1).squeeze()  # (B,)

        sampled_target_seq_tokens = (
            sampled_tokens[:, 1 : l + 1].cpu().detach()
        )  # (B, l)

        name = os.path.splitext(os.path.basename(pdb_path))[0]

        if index_offset:
            with open(output_path, "a", encoding="utf-8") as f:
                for i in range(B):
                    sampled_target_seq = "".join(
                        self.alphabet.get_tok(aa) for aa in sampled_target_seq_tokens[i]
                    )
                    f.write(f">id={i+index_offset}\n")
                    f.write(f"{sampled_target_seq}\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    f">template={name}, T={temperature}, num_res={len(redesigned_residues_positions)}, batch_size={B}\n"
                )
                f.write(f"{native_seqs[target_chain_id]}\n")
                for i in range(B):
                    sampled_target_seq = "".join(
                        self.alphabet.get_tok(aa) for aa in sampled_target_seq_tokens[i]
                    )
                    f.write(f">id={i}\n")
                    f.write(f"{sampled_target_seq}\n")
