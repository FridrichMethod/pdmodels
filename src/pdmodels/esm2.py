from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmForMaskedLM

from pdmodels.globals import AA_ALPHABET, AA_DICT, CHAIN_ALPHABET
from pdmodels.types import Device, ScoreDict
from pdmodels.utils import clean_gpu_cache


class ESM2(nn.Module):
    """ESM2 model for scoring complex structures."""

    def __init__(
        self,
        model_name: Literal[
            "facebook/esm2_t36_3B_UR50D", "facebook/esm2_t33_650M_UR50D"
        ],
        device: Device = None,
    ) -> None:
        """Initialize the ESM2 model."""
        super().__init__()

        self.model_name = model_name
        self._device = device

        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self) -> EsmForMaskedLM:
        """Load the ESM2 model from the transformers library."""
        model = EsmForMaskedLM.from_pretrained(self.model_name)
        model = model.to(self._device)  # type: ignore
        model = model.eval()

        return model

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer for the ESM2 model."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    @property
    def device(self) -> Device:
        """Return the device on which the model is loaded."""
        return next(self.model.parameters()).device

    @clean_gpu_cache
    @torch.no_grad()
    def score(
        self,
        seqs_list: Sequence[str],
        padding_length: int = 10,
        verbose: bool = False,
    ) -> ScoreDict:
        """Calculate the scores of a complex sequence by ESM2 model.

        Mask every single position of the sequence and calculate the entropy of the masked token.

        Args
        ----
        seqs_list: Sequence[str]
            A list of sequences of complex to score towards the given complex structure.
            Chains should be separated by colons.
        padding_length: int
            Padding length for chain separation.
        verbose: bool
            Print the loss and perplexity of the complex.


        Returns
        -------
        output_dict: ScoreDict
            entropy: torch.Tensor[B, L, 20]
                -log{logits} of the masked token at each position.
            loss: torch.Tensor[B, L]
                Cross entropy of the true residue at each position.
            perplexity: torch.Tensor[B,]
                exp{average entropy} of the full sequence.
        """

        aa_tokens = self.tokenizer.encode(
            AA_ALPHABET, add_special_tokens=False, return_tensors="pt"
        ).squeeze()  # (20,)
        padding = (
            self.tokenizer.eos_token  # eos token of last sequence
            + self.tokenizer.mask_token * (padding_length - 2)
            + self.tokenizer.cls_token  # cls token of next sequence
        )
        seq_list_list: list[list[str]] = [seqs.split(":") for seqs in seqs_list]

        # concatenate all sequences
        all_seqs = self.tokenizer(
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
        masked_inputs[all_indices] = self.tokenizer.mask_token_id
        masked_inputs = masked_inputs.to(self.device)

        logits = self.model(masked_inputs).logits.cpu()  # (B * L, L', C)
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

        output_dict: ScoreDict = {
            "entropy": entropy,
            "loss": loss,
            "perplexity": perplexity,
        }

        return output_dict
