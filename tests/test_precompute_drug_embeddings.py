import os
import torch
import tempfile
import pandas as pd
from utils.save_utils import save_mapping
from scripts.precompute_drug_embeddings import main as compute_embeddings

def test_precompute_drug_embeddings():
    # Create a dummy SMILES file
    smiles = ["CCO", "CCC", "C1CCCCC1"]
    df = pd.DataFrame({"Ligand_SMILES": smiles})

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = os.path.join(tmpdir, "drugs.tsv")
        output_pathh = os.path.join(tmpdir, "drug_emb.pt")

        df.to_csv(tsv_path, sep="\t", index=False)

        # Simulate CLI args
        class Args:
            input_path = tsv_path
            output_path = output_pathh
            smiles_column = "Ligand_SMILES"
            input_dim = 7
            edge_dim = 3
            batch_size = 2
            hidden_dim = 32
            output_dim = 16
            num_layers = 2
            heads = 1
            dropout = 0.1

        compute_embeddings(Args)

        # Check that embeddings were saved
        assert os.path.exists(output_pathh)
        loaded = torch.load(output_pathh)
        embeddings = loaded["embeddings"]
        assert embeddings.shape == (len(smiles), Args.output_dim)

        # Check mapping file
        map_path = output_pathh.replace(".pt", "_seq_to_idx.json")
        assert os.path.exists(map_path)

    print("test_precompute_drug_embeddings passed.")
