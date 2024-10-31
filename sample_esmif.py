import os
import sys

import numpy as np
import torch
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import esm
from esm.inverse_folding.multichain_util import (
    _concatenate_coords,
    extract_coords_from_complex,
)
from esm.inverse_folding.util import CoordBatchConverter, load_structure

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_DICT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

torch.backends.cuda.matmul.allow_tf32 = True


def sample_complex(
    model,
    alphabet,
    pdbfile: str,
    output_path: str,
    target_chain_id: str = "A",
    batch_size: int = 1,
    redesigned_residues: str = "",
    omit_aa: str = "",
    temperature: float = 1.0,
    padding_length: int = 10,
    index_offset: int = 0,
):

    device = next(model.parameters()).device

    struct = load_structure(pdbfile)
    native_coords, native_seqs = extract_coords_from_complex(struct)
    all_coords = _concatenate_coords(native_coords, target_chain_id, padding_length=padding_length)

    mask_idx = alphabet.get_idx("<mask>")
    cath_idx = alphabet.get_idx("<cath>")
    pad_idx = alphabet.get_idx("<pad>")

    # Mask out redesigned residues of the target sequence
    target_seq_tokens = torch.tensor(alphabet.encode(native_seqs[target_chain_id]))
    redesigned_residues_positions = torch.tensor([int(i) - 1 for i in redesigned_residues.split()])
    target_seq_tokens[redesigned_residues_positions] = mask_idx

    B = batch_size
    L = all_coords.shape[0]  # length of the whole complex sequence with padding
    l = target_seq_tokens.shape[0]  # length of the target sequence

    # Start with prepend token, followed by the target sequence with masked redesigned residues, and padding to the end
    sampled_tokens = torch.full((B, L + 1), pad_idx, dtype=torch.long)  # (B, L + 1)
    sampled_tokens[:, 0] = cath_idx
    sampled_tokens[:, 1 : l + 1] = target_seq_tokens
    sampled_tokens = sampled_tokens.to(device)

    # Tokens to remove from the output distribution
    removed_aa_tokens = torch.tensor(
        alphabet.encode(
            "".join(aa for aa in alphabet.all_toks if (aa not in AA_ALPHABET) or (aa in omit_aa))
        )
    )

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(all_coords, None, None) for _ in range(B)]
    batch_coords, confidence, _, _, padding_mask = batch_converter(batch, device=device)

    # Save incremental states for faster sampling
    incremental_state = {}

    with torch.no_grad():
        # Run encoder only once
        encoder_out = model.encoder(batch_coords, padding_mask, confidence)

        # Autoregressively decode the sequence one token at a time
        for j in range(1, l + 1):
            logits, _ = model.decoder(
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

    sampled_target_seq_tokens = sampled_tokens[:, 1 : l + 1].cpu().detach()  # (B, l)

    name = os.path.splitext(os.path.basename(pdbfile))[0]

    if index_offset:
        with open(output_path, "a") as f:
            for i in range(B):
                sampled_target_seq = "".join(
                    alphabet.get_tok(aa) for aa in sampled_target_seq_tokens[i]
                )
                f.write(f">id={i+index_offset}\n")
                f.write(f"{sampled_target_seq}\n")
    else:
        with open(output_path, "w") as f:
            f.write(
                f">template={name}, T={temperature}, num_res={len(redesigned_residues_positions)}, batch_size={B}\n"
            )
            f.write(f"{native_seqs[target_chain_id]}\n")
            for i in range(B):
                sampled_target_seq = "".join(
                    alphabet.get_tok(aa) for aa in sampled_target_seq_tokens[i]
                )
                f.write(f">id={i}\n")
                f.write(f"{sampled_target_seq}\n")
