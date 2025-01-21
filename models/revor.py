import os
import pickle
import tempfile
from collections import deque
from typing import Callable, Self

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from matplotlib.lines import Line2D

from models.esmif import ESMIF
from models.mpnn import MPNN
from models.utils import count_mutations, seqs_list_to_tensor, tensor_to_seqs_list


class ReVor:
    """Reversed evolution for backward Monte Carlo sampling of largely mutated sequences."""

    _NODE_COLORS: dict[str, str] = {
        "start": "green",
        "intermediate": "skyblue",
        "end": "red",
    }

    def __init__(
        self, model: MPNN | ESMIF, pdb_path: str, seq_wt: str, **kwargs
    ) -> None:
        """Initialize the ReVor model."""
        self.model = model
        self.pdb_path = pdb_path
        self.seqs_wt = seq_wt
        self.kwargs = kwargs

        self.dag: nx.DiGraph = nx.DiGraph()
        self.q: deque[str] = deque()
        self.it: int = 0

        self._aa_wt: torch.Tensor | None = None
        self._chain_breaks: list[int] | None = None

    @property
    def aa_wt(self) -> torch.Tensor:
        """Wild-type amino acid sequence encoded as a tensor."""
        if self._aa_wt is None:
            self._aa_wt = seqs_list_to_tensor([self.seqs_wt]).squeeze(0)
        return self._aa_wt

    @property
    def chain_breaks(self) -> list[int]:
        """List of length of each chain in the wild-type sequence."""
        if self._chain_breaks is None:
            self._chain_breaks = [len(seq_wt) for seq_wt in self.seqs_wt.split(":")]
        return self._chain_breaks

    def _save_checkpoint(self, checkpoint_path: str) -> None:
        """Save the checkpoint of the ReVor model."""
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)

        print(f"Saving checkpoint to {checkpoint_path}")
        checkpoint = {
            "dag": self.dag,
            "q": self.q,
            "it": self.it,
        }

        # Atomic write the checkpoint file
        try:
            with tempfile.NamedTemporaryFile(dir=dir_name, delete=False) as temp_file:
                temp_path = temp_file.name
                pickle.dump(checkpoint, temp_file, pickle.HIGHEST_PROTOCOL)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, checkpoint_path)
        except Exception as e:
            os.remove(temp_path)
            raise RuntimeError(
                f"Failed to save checkpoint to {checkpoint_path}: {e}"
            ) from e

    def load(self, checkpoint_path: str) -> Self:
        """Load the checkpoint of the ReVor model."""
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.dag = checkpoint["dag"]
        self.q = checkpoint["q"]
        self.it = checkpoint["it"]

        return self

    def _score(self, seqs_list: list[str], repeat: int) -> torch.Tensor:
        """Calculate the average loss for a batch of sequences."""
        seqs_num = len(seqs_list)

        batch_seqs_list = seqs_list * repeat
        batch_output_dict = self.model.score(
            self.pdb_path, batch_seqs_list, **self.kwargs
        )
        batch_entropy = batch_output_dict["entropy"]  # (B * R, L, 20)
        batch_loss = batch_output_dict["loss"]  # (B * R, L)

        entropy = batch_entropy.view(repeat, seqs_num, -1, 20).mean(dim=0)  # (B, L, 20)
        loss = batch_loss.view(repeat, seqs_num, -1).mean(dim=0)  # (B, L)
        target = self.aa_wt.repeat(seqs_num, 1)  # (B, L)
        loss_wt = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)

        score = loss - loss_wt

        return score

    def _iterate(
        self,
        seqs_list: list[str],
        cutoff: float,
        repeat: int,
        max_step: int,
        n_samples: int,
        temperature: float,
    ) -> None:
        """Iterate over the sequences and revert the mutations that increase the loss."""

        self.it += 1

        aa_tensor = seqs_list_to_tensor(seqs_list)
        mutation_mask = aa_tensor != self.aa_wt

        # Calculate the score for reverting the mutations
        score = self._score(seqs_list, repeat)
        score[~mutation_mask] = -np.inf
        revert_values, revert_indices = torch.topk(score, max_step)
        revert_mask = torch.zeros_like(aa_tensor, dtype=torch.bool).scatter(
            1, revert_indices, revert_values > cutoff
        )
        score[~revert_mask] = -np.inf

        # Filter out the sequences that satisfy the cutoff
        terminate_mask = revert_mask.any(dim=1)
        seqs_list_filtered = []
        for seqs, mask in zip(seqs_list, terminate_mask.tolist()):
            if mask:
                seqs_list_filtered.append(seqs)
            else:
                self.dag.nodes[seqs]["role"] = "end"
        if not seqs_list_filtered:
            return

        seqs_num_filtered = len(seqs_list_filtered)
        aa_tensor_filtered = aa_tensor[terminate_mask]  # (B', L)
        score_filtered = score[terminate_mask]  # (B', L)

        # Sample the mutations to revert
        prob = (
            torch.sigmoid(score_filtered / temperature)
            .unsqueeze(1)
            .repeat(1, n_samples, 1)
        )  # (B', N, L)
        sample_mask = torch.bernoulli(prob).bool()
        while not torch.all((resample_mask := sample_mask.any(dim=(1, 2)))):
            sample_mask[~resample_mask] = torch.bernoulli(prob[~resample_mask]).bool()
        new_aa_tensor = torch.where(
            sample_mask, self.aa_wt, aa_tensor_filtered.unsqueeze(1)
        ).view(
            seqs_num_filtered * n_samples, -1
        )  # (B' * N, L)

        # Update the graph with the reverted sequences
        new_seqs_list = tensor_to_seqs_list(new_aa_tensor, self.chain_breaks)
        for i, new_seqs in enumerate(new_seqs_list):
            old_seqs = seqs_list_filtered[i // n_samples]
            if new_seqs == old_seqs:
                continue
            if new_seqs not in self.dag:
                self.q.append(new_seqs)
                self.dag.add_node(
                    new_seqs,
                    iteration=self.it,
                    distance=count_mutations(new_seqs, self.seqs_wt).item(),
                    role="intermediate",
                )
            self.dag.add_edge(
                old_seqs,
                new_seqs,
                weight=count_mutations(new_seqs, old_seqs).item(),
            )

    def revert(
        self,
        input_path: str,
        cutoff: float = 0.1,
        batch_size: int = 32,
        repeat: int = 4,
        max_step: int = 3,
        n_samples: int = 6,
        temperature: float = 1.0,
        checkpoint_path: str = "",
        save_checkpoint_interval: int = 20,
    ) -> None:
        """Revert the mutations that increase the loss.

        Args:
        -----
        input_path: str
            Path to the FASTA file containing the mutated sequences,
            or the checkpoint file of the ReVor model.
        cutoff: float
            Cutoff value for delta loss to revert the mutations.
        batch_size: int
            Number of sequences to process in each batch.
        repeat: int
            Number of duplicates for each sequence in the batch to average the loss.
        max_step: int
            Maximum number of mutations to revert in each iteration.
        n_samples: int
            Number of sequences to sample for reverting the mutations.
        temperature: float
            Temperature for sampling the mutations.
        checkpoint_path: str
            Path to save the checkpoint of the ReVor model.
            If not provided, the checkpoint is not saved.
        save_checkpoint_interval: int
            Interval to save the checkpoint of the ReVor model in iterations.

        Notes:
        -----
        - The actual batch size is `batch_size * repeat`.
        """

        if input_path.endswith(".pkl"):
            self.load(input_path)
        elif input_path.endswith(".fasta"):
            with open(input_path) as f:
                for title, seqs in SimpleFastaParser(f):
                    self.q.append(seqs)
                    self.dag.add_node(
                        seqs,
                        title=title,
                        iteration=self.it,
                        distance=count_mutations(seqs, self.seqs_wt).item(),
                        role="start",
                    )
        else:
            raise ValueError(
                "Input file must be either a FASTA file or a checkpoint file."
            )

        seqs_list: list[str] = []
        while True:
            if len(seqs_list) < batch_size:
                if self.q:
                    seqs = self.q.popleft()
                    seqs_list.append(seqs)
                    continue
                else:
                    if not seqs_list:
                        break

            print(f"Iteration {self.it}")
            print(f"Number of sequences remaining: {len(self.q) + len(seqs_list)}")

            self._iterate(
                seqs_list,
                cutoff,
                repeat,
                max_step,
                n_samples,
                temperature,
            )
            seqs_list.clear()

            if checkpoint_path and not self.it % save_checkpoint_interval:
                self._save_checkpoint(checkpoint_path)

        for topology, nodes in enumerate(nx.topological_generations(self.dag)):
            for node in nodes:
                self.dag.nodes[node]["topology"] = topology

        assert all(
            data["role"] == "end"
            for node, data in self.dag.nodes(data=True)
            if self.dag.out_degree(node) == 0
        )
        assert all(
            data["role"] == "start"
            for node, data in self.dag.nodes(data=True)
            if self.dag.in_degree(node) == 0 and self.dag.out_degree(node) > 0
        )
        assert all(
            data["role"] != "end"
            for node, data in self.dag.nodes(data=True)
            if self.dag.in_degree(node) > 0 and self.dag.out_degree(node) > 0
        )

    def _get_subdag(self, titles: list[str]) -> nx.DiGraph:
        """Get the subgraph of the reverted sequences starting from the given title."""
        reachable_nodes = set()
        for title in titles:
            root = next(
                (
                    node
                    for node, data in self.dag.nodes(data=True)
                    if data.get("title") == title
                ),
                None,
            )
            if root is None:
                raise ValueError(f"No node found with title '{title}'.")
            reachable_nodes |= nx.descendants(self.dag, root) | {root}

        subdag = self.dag.subgraph(reachable_nodes)

        return subdag

    def plot(
        self,
        layout: Callable,
        subdag_titles: str | list[str] | None = None,
        figsize: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        """Plot the graph of the reverted sequences.

        Args:
        -----
        layout: Callable
            Layout function to plot the graph.
        subdag_titles: str | list[str]
            Title of the starting sequence to plot the subgraph.
        figsize: tuple[int, int]
            Size of the figure.
        kwargs: dict
            Additional keyword arguments for the layout function.
        """

        if subdag_titles is None:
            subdag = self.dag
        elif isinstance(subdag_titles, str):
            subdag = self._get_subdag([subdag_titles])
        else:
            subdag = self._get_subdag(subdag_titles)

        plt.figure(figsize=figsize)

        pos = layout(subdag, **kwargs)
        node_color = [
            self._NODE_COLORS[data["role"]] for _, data in subdag.nodes(data=True)
        ]
        nx.draw(
            subdag,
            pos,
            node_size=100,
            node_color=node_color,
            edge_color="gray",
            alpha=0.5,
        )

        node_labels = nx.get_node_attributes(subdag, "title")
        nx.draw_networkx_labels(
            subdag, pos, labels=node_labels, font_size=8, font_weight="bold"
        )
        edge_labels = nx.get_edge_attributes(subdag, "weight")
        nx.draw_networkx_edge_labels(subdag, pos, edge_labels=edge_labels, font_size=8)

        plt.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=role,
                    markerfacecolor=color,
                    markersize=10,
                )
                for role, color in self._NODE_COLORS.items()
            ]
        )
        plt.show()
        plt.close()

    def save(self, output_dir: str) -> None:
        """Save the reverted sequences and the graph."""
        os.makedirs(output_dir, exist_ok=True)

        output_fasta_path = os.path.join(output_dir, "sequences.fasta")
        with open(output_fasta_path, "w") as f:
            i = 0
            for seqs in self.dag:
                if not self.dag.out_degree(seqs):
                    f.write(f">{i}\n{seqs}\n")
                    i += 1

        output_graph_path = os.path.join(output_dir, "graph.pkl")
        with open(output_graph_path, "wb") as f:
            pickle.dump(self.dag, f, pickle.HIGHEST_PROTOCOL)
