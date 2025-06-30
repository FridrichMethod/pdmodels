import argparse
import os
from typing import Sequence

import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from esm.esmfold.v1.misc import batch_encode_sequences, collate_dense_tensors
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from pdmodels.basemodels import TorchModel
from pdmodels.types import Device
from pdmodels.utils import Timer


class EsmForProteinFoldingNew(EsmForProteinFolding):
    """HuggingFace ESMFold model with the original infer method."""

    @torch.no_grad()
    def infer(
        self,
        sequences: str | list[str],
        residx: torch.Tensor | None = None,
        masking_pattern: torch.Tensor | None = None,
        num_recycles: int | None = None,
        residue_index_offset: int | None = 512,
        chain_linker: str | None = "G" * 25,
    ):
        """Rewrite the infer method as the original ESMFold model."""
        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        output = super().forward(
            aatype,
            attention_mask=mask,
            position_ids=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )

        output["atom37_atom_exists"] = output[
            "atom37_atom_exists"
        ] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        return output


class ESMFold(TorchModel):
    """ESMFold model for predicting the 3D structure of a protein from its sequence."""

    def __init__(self, device: Device = None, chunk_size: int | None = None) -> None:
        """Initialize the ESMFold model."""
        super().__init__(device=device)

        self.chunk_size = chunk_size

        self.model, self.tokenizer = self._load_model()

    def _load_model(self) -> tuple[EsmForProteinFoldingNew, AutoTokenizer]:
        """Load the ESMFold model and tokenizer from the transformers library."""
        model = EsmForProteinFoldingNew.from_pretrained("facebook/esmfold_v1")
        model = model.to(self.device)  # type: ignore
        model = model.eval()
        model.trunk.set_chunk_size(self.chunk_size)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

        return model, tokenizer  # type: ignore

    def batch_predict(
        self,
        seqs_list: Sequence[str],
        num_recycles: int = 4,
        residue_index_offset: int = 512,
        chain_linker: str = "G" * 25,
    ) -> dict[str, torch.Tensor]:
        """Predict the 3D structure of proteins from a list of sequences.

        Args:
        -----
        seqs_list: Sequence[str]
            List of protein sequences.
        num_recycles: int
            Number of recycling steps.
        residue_index_offset: int
            Offset for the residue index.
        chain_linker: str
            A string of Gs representing the linker between the two chains.

        Returns:
        --------
        output: dict[str, torch.Tensor]
            Dictionary containing the predicted 3D structure of the protein.
        """

        with torch.no_grad():
            output = self.model.infer(
                seqs_list,
                num_recycles=num_recycles,
                residue_index_offset=residue_index_offset,
                chain_linker=chain_linker,
            )

        return output

    def predict(
        self,
        seqs: str,
        num_recycles: int = 4,
        residue_index_offset: int = 512,
        chain_linker: str = "G" * 25,
    ) -> dict[str, torch.Tensor]:
        """Predict the 3D structure of a protein complex from its sequences.

        Args:
        -----
        seqs: str
            Protein sequences.
        num_recycles: int
            Number of recycling steps.
        residue_index_offset: int
            Offset for the residue index.
        chain_linker: str
            A string of Gs representing the linker between the two chains.

        Returns:
        --------
        output: dict[str, torch.Tensor]
            Dictionary containing the predicted 3D structure of the protein.
        """

        return self.batch_predict(
            [seqs],
            num_recycles=num_recycles,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )

    def save(
        self, output: dict[str, torch.Tensor], output_dir: str, title: str
    ) -> None:
        """Save the predicted 3D structure of the protein to a file.

        Args:
        -----
        output: dict[str, torch.Tensor]
            Dictionary containing the predicted 3D structure of the protein.
        output_dir: str
            Path to the directory where the output files will be saved.
        title: str
            Title of the protein.
        """

        os.makedirs(output_dir, exist_ok=True)

        torch.save(output, os.path.join(output_dir, f"{title}.pt"))
        print(f"Saved {title}.pt to {output_dir}")

        pdb = self.model.output_to_pdb(output)[0]
        print(f"Saved {title}.pdb to {output_dir}")
        with open(os.path.join(output_dir, f"{title}.pdb"), "w") as f:
            f.write(pdb)

    def run(
        self,
        fasta_path: str,
        output_dir: str,
        num_recycles: int = 4,
        residue_index_offset: int = 512,
        chain_linker: str = "G" * 25,
    ) -> None:
        """Predict the 3D structure of proteins from a fasta file containing their sequences.

        Args:
        -----
        fasta_path: str
            Path to the fasta file containing the protein sequences.
        output_dir: str
            Path to the directory where the output files will be saved.
        num_recycles: int
            Number of recycling steps.
        residue_index_offset: int
            Offset for the residue index.
        chain_linker: str
            A string of Gs representing the linker between the two chains.
        """

        os.makedirs(output_dir, exist_ok=True)

        # Faster to parse the whole fasta file at once
        with open(fasta_path) as f:
            records = list(SimpleFastaParser(f))

        for record in tqdm(records):
            title, seqs = record
            output = self.predict(
                seqs,
                num_recycles=num_recycles,
                residue_index_offset=residue_index_offset,
                chain_linker=chain_linker,
            )
            average_plddt = output["mean_plddt"].item()
            average_pae = output["predicted_aligned_error"].mean()
            ptm = output["ptm"].item()
            print(
                f"{title}: Average pLDDT: {average_plddt}, Average PAE: {average_pae}, pTM: {ptm}"
            )
            self.save(output, output_dir, title)


def cli(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esmfold = ESMFold(device=device)
    with Timer() as timer:
        esmfold.run(
            args.fasta_path,
            args.output_dir,
            num_recycles=args.num_recycles,
            residue_index_offset=args.residue_index_offset,
            chain_linker=args.chain_linker,
        )
    print(f"Total time: {timer.elapsed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path",
        type=str,
        help="Path to the fasta file containing the protein sequences.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory where the output files will be saved.",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=4,
        help="Number of recycling steps.",
    )
    parser.add_argument(
        "--residue_index_offset",
        type=int,
        default=512,
        help="Offset for the residue index.",
    )
    parser.add_argument(
        "--chain_linker",
        type=str,
        default="G" * 25,
        help="A string of Gs representing the linker between the two chains.",
    )
    args = parser.parse_args()

    cli(args)


if __name__ == "__main__":
    main()
