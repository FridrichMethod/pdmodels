import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from models.globals import AA_ALPHABET, AA_DICT, CHAIN_ALPHABET


def score_complex(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    seqs_list: list[str],
    padding_length: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the scores of a complex sequence by ESM2 model.

    Mask every single position of the sequence and calculate the entropy of the masked token.

    Args:
        model (EsmForMaskedLM):
            esm2 model, either `3B` or `650M`.
        tokenizer (AutoTokenizer):
            Tokenizer for the model.
        seqs_list (list[str]):
            A list of sequences of complex to score towards the given complex structure.
            Chains should be separated by colons.
        padding_length (int):
            Padding length for chain separation.
        verbose (bool):
            Print the loss and perplexity of the complex.

    Returns:
        (entropy, loss, perplexity) (tuple):
            - entropy (torch.Tensor (B, L, 20)): -log{logits} of the masked token at each position.
            - loss (torch.Tensor (B, L)): Cross entropy of the true residue at each position.
            - perplexity (torch.Tensor (B,)): exp{average entropy} of the full sequence.
    """

    device = next(model.parameters()).device

    aa_tokens = tokenizer.encode(
        AA_ALPHABET, add_special_tokens=False, return_tensors="pt"
    ).squeeze()  # (20,)
    padding = (
        tokenizer.eos_token  # eos token of last sequence
        + tokenizer.mask_token * (padding_length - 2)
        + tokenizer.cls_token  # cls token of next sequence
    )
    seq_list_list: list[list[str]] = [seqs.split(":") for seqs in seqs_list]

    # concatenate all sequences
    all_seqs = tokenizer(
        [padding.join(seq_list) for seq_list in seq_list_list], return_tensors="pt"
    )[
        "input_ids"
    ]  # (B, L')
    all_masks = torch.isin(all_seqs, aa_tokens)  # (B, L')
    seq_len = torch.sum(all_masks[0]).item()

    # mask every single position of the sequence
    masked_inputs = all_seqs.repeat_interleave(seq_len, dim=0)  # (B * L, L')
    all_indices: tuple[torch.Tensor, torch.Tensor] = (
        torch.arange(masked_inputs.shape[0]),  # (B * L,)
        torch.where(all_masks)[1],  # (B * L,)
    )
    masked_inputs[all_indices] = tokenizer.mask_token_id
    masked_inputs = masked_inputs.to(device)

    with torch.no_grad():
        logits = model(masked_inputs).logits
    logits = logits.cpu().detach()  # (B * L, L', C)
    entropy = -(
        logits[all_indices][:, aa_tokens].view(-1, seq_len, 20).log_softmax(dim=-1)
    )  # (B, L, 20)

    target = torch.tensor(
        [[AA_DICT[aa] for aa in "".join(seq_list)] for seq_list in seq_list_list]
    )  # (B, L)
    loss = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)
    perplexity = torch.exp(loss.mean(dim=-1))  # (B,)

    if verbose:
        for l, seq_list in enumerate(seq_list_list):
            print(f"target_seq: {seqs_list[l]}")
            print("loss:")
            k = 0
            for i, chain in enumerate(seq_list):
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

    esm2_model = EsmForMaskedLM.from_pretrained(args.model)
    esm2_model = esm2_model.to(device)
    esm2_model = esm2_model.eval()
    esm2_tokenizer = AutoTokenizer.from_pretrained(args.model)

    entropy, loss, perplexity = score_complex(
        esm2_model,
        esm2_tokenizer,
        args.seqs,
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
    parser.add_argument("model", type=str, help="Model name.")
    parser.add_argument(
        "seqs", type=str, help="Sequences of a complex structure separated by colons."
    )
    parser.add_argument("output_path", type=str, help="Path to save the output.")
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
