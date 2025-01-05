import argparse
import os
from typing import Optional, Union

import esm
import esm.esmfold.v1.esmfold
import torch
from Bio import SeqIO
from tqdm.auto import tqdm


class ESMFold:
    """ESMFold model for predicting the 3D structure of a protein from its sequence."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.device = device
        self.chunk_size = chunk_size

        self.model = self._load_model()

    def _load_model(self) -> esm.esmfold.v1.esmfold.ESMFold:
        model = esm.pretrained.esmfold_v1()
        model.to(self.device)
        model.eval()

        # This chunk size was set to reduce memory usage
        model.set_chunk_size(self.chunk_size)

        return model

    def to(self, device: str | torch.device | None) -> None:
        """Move the model to the given device."""
        self.device = device
        self.model = self.model.to(self.device)  # type: ignore

    def predict(self, seq: str) -> dict[str, torch.Tensor]:
        """Predict the 3D structure of a protein from its sequence.

        Args:
        -----
        seq: str
            Protein sequence.

        Returns:
        --------
        output: dict[str, torch.Tensor]
            Dictionary containing the predicted 3D structure of the protein.
        """

        with torch.no_grad():
            output = self.model.infer([seq])  # Faster to run seqs one by one

        return output

    def __call__(self, fasta_path: str, output_dir: str) -> None:
        """Predict the 3D structure of proteins from a fasta file containing their sequences.

        Args:
        -----
        fasta_path: str
            Path to the fasta file containing the protein sequences.
        output_dir: str
            Path to the directory where the output files will be saved.
        """

        os.makedirs(output_dir, exist_ok=True)

        # Faster to parse the whole fasta file at once
        records = [
            (record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")
        ]

        for record in tqdm(records):
            title, seq = record

            output = self.predict(seq)
            average_plddt = output["mean_plddt"].item()
            average_pae = output["predicted_aligned_error"].mean()
            ptm = output["ptm"].item()
            print(
                f"{title}: Average pLDDT: {average_plddt}, Average PAE: {average_pae}, pTM: {ptm}"
            )
            torch.save(output, os.path.join(output_dir, f"{title}.pt"))

            pdb = self.model.output_to_pdb(output)[0]
            print(f"Saved {title}.pdb to {output_dir}")
            with open(os.path.join(output_dir, f"{title}.pdb"), "w") as f:
                f.write(pdb)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esmfold = ESMFold(device=device)
    esmfold(args.fasta_path, args.output_dir)


if __name__ == "__main__":
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
    args = parser.parse_args()

    main(args)
