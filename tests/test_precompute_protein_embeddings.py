import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


import pytest
import torch
import pandas as pd
import tempfile
import argparse
from scripts import precompute_protein_embeddings as prep
from unittest.mock import MagicMock
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

def test_chunked():
    data = list(range(10))
    chunks = list(prep.chunked(data, 3))
    assert chunks == [[0,1,2],[3,4,5],[6,7,8],[9]]

def test_load_dataset():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
        tmp.write("Ligand SMILES\tBindingDB Target Chain Sequence\tbinary_label\n")
        tmp.write("CCC\tSEQA\t1\n")
        tmp.write("CCO\tSEQB\t0\n")
        tmp.write("CCC\tSEQA\t1\n")  # duplicate sequence
        tmp_path = tmp.name

    sequences = prep.load_dataset(tmp_path)
    assert set(sequences) == {"SEQA", "SEQB"}

def test_save_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/embeddings.pt"
        embeddings = torch.rand(2, 1280)
        sequence_ids = ["SEQ1", "SEQ2"]

        prep.save_embeddings(embeddings, sequence_ids, output_path)

        saved = torch.load(output_path)
        assert "embeddings" in saved and "ids" in saved
        assert saved["embeddings"].shape == (2, 1280)
        assert saved["ids"] == sequence_ids

def test_main_mocked(monkeypatch):
    # Mock ESMEncoder to avoid GPU and HF call
    class DummyEncoder:
        def __call__(self, sequences):
            return torch.ones(len(sequences), 1280)

    monkeypatch.setattr(prep, "ESMEncoder", lambda *args, **kwargs: DummyEncoder())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
        tmp.write("Ligand SMILES\tBindingDB Target Chain Sequence\tbinary_label\n")
        tmp.write("CCC\tSEQA\t1\nCCC\tSEQB\t0\n")
        tmp_path = tmp.name

    with tempfile.TemporaryDirectory() as tmpdir:
        args = argparse.Namespace(
            input_path=tmp_path,
            output_path=f"{tmpdir}/output.pt",
            sequence_column="BindingDB Target Chain Sequence",
            batch_size=1,
            model_name="facebook/esm2_t33_650M_UR50D"
        )
        prep.main(args)

        data = torch.load(args.output_path)
        assert data["embeddings"].shape[0] == 2
        assert set(data["ids"]) == {"SEQA", "SEQB"}
