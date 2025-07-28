import sys
import os
import tempfile
import torch
import pandas as pd
import json
from unittest.mock import patch, MagicMock
import argparse
import pytest

# Path setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts import precompute_protein_embeddings as prep


def test_chunked():
    data = list(range(10))
    chunks = list(prep.chunked(data, 4))
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_load_dataset():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
        tmp.write("BindingDB_Target_Chain_Sequence\n")
        tmp.write("SEQ_A\nSEQ_B\nSEQ_A\n")  # duplicate
        tmp_path = tmp.name

    sequences = prep.load_dataset(tmp_path)
    assert set(sequences) == {"SEQ_A", "SEQ_B"}


def test_full_pipeline_with_mock():
    sequences = ["SEQ_A", "SEQ_B", "SEQ_C"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = os.path.join(tmpdir, "mock_proteins.tsv")
        out_path = os.path.join(tmpdir, "protein_embeddings.pt")

        # Write dummy input TSV
        df = pd.DataFrame({"BindingDB_Target_Chain_Sequence": sequences})
        df.to_csv(tsv_path, sep="\t", index=False)

        # Mock the ESMEncoder forward to return fake embeddings
        fake_embedding = torch.rand(len(sequences), 320)

        class DummyEncoder:
            def __init__(self, *args, **kwargs):
                self.model = MagicMock()
                self.model.config.hidden_size = 320

            def __call__(self, batch):
                return fake_embedding[:len(batch)]

        with patch("scripts.precompute_protein_embeddings.ESMEncoder", DummyEncoder):
            args = argparse.Namespace(
                input_path=tsv_path,
                output_path=out_path,
                sequence_column="BindingDB_Target_Chain_Sequence",
                model_name="facebook/esm2_t6_8M_UR50D",
                batch_size=2
            )
            prep.main(args)

        # Validate .pt file
        assert os.path.exists(out_path)
        loaded = torch.load(out_path)
        assert "embeddings" in loaded and "ids" in loaded
        assert loaded["embeddings"].shape == (len(sequences), 320)
        assert set(loaded["ids"]) == set(sequences)

        # Validate mapping JSON
        map_path = out_path.replace(".pt", "_seq_to_idx.json")
        assert os.path.exists(map_path)
        with open(map_path) as f:
            mapping = json.load(f)
        assert set(mapping.keys()) == set(sequences)
