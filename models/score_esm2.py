import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from .globals import AA_ALPHABET, AA_DICT, CHAIN_ALPHABET


def score_complex(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    seqs: str,
    padding_length: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Calculate the scores of a complex sequence by ESM2 model.

    Mask every single position of the sequence and calculate the entropy of the masked token.

    Parameters:
    -----------
    model: EsmForMaskedLM
        esm2 model, either 3B or 650M.
    tokenizer: AutoTokenizer
        Tokenizer for the model.
    seqs: str
        Sequences of complex to score towards the given complex structure.
        Chains should be separated by colons.
    padding_length: int
        Padding length for chain separation.
    verbose: bool
        Print the loss and perplexity of the complex.

    Returns:
    --------
    entropy: torch.Tensor (L, 20)
        -log{logits} of the masked token at each position.
    loss: torch.Tensor (L,)
        Cross entropy of the true residue at each position.
    perplexity: float
        exp{average entropy} of the full sequence.
    """

    device = next(model.parameters()).device

    aa_tokens = tokenizer.encode(
        AA_ALPHABET, add_special_tokens=False, return_tensors="pt"
    ).squeeze()
    padding = (
        tokenizer.cls_token
        + tokenizer.mask_token * (padding_length - 2)
        + tokenizer.eos_token
    )
    seq_list = seqs.split(":")

    all_seqs = tokenizer(padding.join(seq_list), return_tensors="pt")[
        "input_ids"
    ].squeeze()
    all_indices = torch.where(all_seqs.unsqueeze(-1) == aa_tokens)[0]

    # mask every single position of the sequence
    mask = torch.eye(len(all_seqs), dtype=torch.long)
    masked_inputs = (all_seqs * (1 - mask) + tokenizer.mask_token_id * mask)[
        all_indices
    ]
    masked_inputs = masked_inputs.to(device)

    with torch.no_grad():
        logits = model(masked_inputs).logits
    logits = logits.cpu().detach()
    entropy = -(
        logits[torch.arange(logits.shape[0]), all_indices][:, aa_tokens].log_softmax(
            dim=-1
        )
    )  # (L, 20)

    target = torch.tensor([AA_DICT[aa] for aa in "".join(seq_list)])
    loss = torch.gather(entropy, 1, target.unsqueeze(1)).squeeze()  # (L,)
    perplexity = torch.exp(loss.mean()).item()  # scalar

    if verbose:
        print("loss:")
        k = 0
        for i, chain in enumerate(seq_list):
            loss_chunk = loss[k : k + len(chain)]
            k += len(chain)
            print(f"chain {CHAIN_ALPHABET[i]}")
            for j, (aa, loss_val) in enumerate(zip(chain, loss_chunk)):
                print(f"{aa}{j+1}: {loss_val.item()}")
        print(f"perplexity: {perplexity}")

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
