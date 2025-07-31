import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


import pytest
from data.utils.featurize_mol import mol_to_graph_data_obj
from rdkit import Chem
from torch_geometric.data import Data


def test_valid_smiles():
    smiles = "CCO"  # ethanol
    data = mol_to_graph_data_obj(smiles)

    assert isinstance(data, Data)
    assert data.x.shape[0] == 3  # 3 atoms: C, C, O
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.smiles == smiles


def test_bidirectional_edges():
    smiles = "CCO"
    data = mol_to_graph_data_obj(smiles)

    # Should be 2 edges for each bond (bidirectional)
    mol = Chem.MolFromSmiles(smiles)
    num_bonds = mol.GetNumBonds()
    assert data.edge_index.shape[1] == 2 * num_bonds


def test_atom_feature_dim():
    smiles = "CCO"
    data = mol_to_graph_data_obj(smiles)
    assert data.x.shape[1] == 7  # 7 atom features as defined


def test_invalid_smiles():
    with pytest.raises(ValueError):
        mol_to_graph_data_obj("NotASmiles")


def test_ring_structure():
    # Benzene ring: C1=CC=CC=C1
    data = mol_to_graph_data_obj("C1=CC=CC=C1")
    assert data.x.shape[0] == 6
    assert data.edge_index.shape[1] == 12  # 6 bonds × 2 (bidirectional)