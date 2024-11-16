import os
import sys


def test_packages():
    import esm
    import numpy as np
    import pandas as pd
    import torch
    import torch_geometric
    import torch_sparse
    from torch_geometric.nn import MessagePassing
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, EsmForMaskedLM


def test_models():
    from models.sample_esmif import sample_complex
    from models.score_esm2 import score_complex as score_complex_esm2
    from models.score_esmif import score_complex as score_complex_esmif
    from models.score_ligandmpnn import LigandMPNNBatch
    from models.score_ligandmpnn import score_complex as score_complex_ligandmpnn


def test_cuda():
    import torch

    if not torch.cuda.is_available():
        raise Exception("CUDA not available")


def test_ligandmpnn():
    import torch

    from models.score_ligandmpnn import LigandMPNNBatch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        "./models/model_params/ligandmpnn_v_32_020_25.pt",
        map_location=device,
        weights_only=True,
    )
    ligandmpnn = LigandMPNNBatch(
        ligand_mpnn_use_side_chain_context=True,
        device=device,
    )
    ligandmpnn.load_state_dict(checkpoint["model_state_dict"])
    ligandmpnn.eval().to(device)


def test_esm2_650M():
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm2_650M_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm2_650M_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm2_650M_model = esm2_650M_model.eval().to(device)


def test_esmif():
    import esm
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esmif_model, esmif_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esmif_model = esmif_model.eval().to(device)


def test_esm3B():
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_3B_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
    esm_3B_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    esm_3B_model = esm_3B_model.eval().to(device)


def main():
    print("Testing environment...")
    test_packages()
    test_models()
    test_cuda()

    for test_func in [test_ligandmpnn, test_esm2_650M, test_esmif, test_esm3B]:
        print(f"Running {test_func.__name__}")
        try:
            test_func()
        except Exception as e:
            print(f"Error in {test_func.__name__}: {e}")
            raise

    print("All tests passed!")


if __name__ == "__main__":
    main()
