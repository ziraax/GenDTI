"""
Test script for the reward function module.
Tests ADMET scoring and reward computation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.reward import RewardFunction, ADMETScorer
from models.graph_diffusion import create_molecule_graph

def test_admet_scorer():
    """Test ADMET scoring functionality."""
    print("🧪 Testing ADMET Scorer")
    print("=" * 50)
    
    scorer = ADMETScorer()
    
    # Test molecules with known properties
    test_molecules = [
        ("CCO", "Ethanol - simple, drug-like"),
        ("c1ccccc1", "Benzene - aromatic"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin - actual drug"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine - stimulant"),
        ("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "Long alkane - not drug-like"),
        ("C", "Methane - too simple"),
        ("INVALID_SMILES", "Invalid SMILES"),
        ("", "Empty string")
    ]
    
    for smiles, description in test_molecules:
        score = scorer.score_molecule(smiles)
        print(f"  {description}")
        print(f"    SMILES: {smiles}")
        print(f"    ADMET Score: {score:.3f}")
        print()
    
    print("✅ ADMET Scorer test complete!\n")

def test_reward_function_mock():
    """Test reward function with a mock DTI model."""
    print("🧪 Testing Reward Function (Mock DTI)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a mock DTI model that returns random scores
    class MockDTIModel(torch.nn.Module):
        def forward(self, mol_batch, protein_embeddings):
            batch_size = len(mol_batch.to_data_list())
            return torch.rand(batch_size, device=device)
    
    mock_dti_model = MockDTIModel().to(device)
    
    # Initialize reward function
    reward_function = RewardFunction(
        dti_model=mock_dti_model,
        beta_dti=1.0,
        beta_admet=1.0,
        device=device
    )
    
    # Test molecules
    test_smiles = [
        "CCO",                              # Ethanol
        "c1ccccc1",                         # Benzene
        "CC(=O)OC1=CC=CC=C1C(=O)O",        # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"     # Caffeine
    ]
    
    # Convert SMILES to graphs
    molecule_graphs = []
    valid_smiles = []
    
    for smiles in test_smiles:
        graph = create_molecule_graph(smiles)
        if graph is not None:
            molecule_graphs.append(graph)
            valid_smiles.append(smiles)
            print(f"  ✅ Successfully converted: {smiles}")
        else:
            print(f"  ❌ Failed to convert: {smiles}")
    
    if len(molecule_graphs) == 0:
        print("❌ No valid molecules to test!")
        return
    
    # Create mock protein embeddings
    batch_size = len(molecule_graphs)
    protein_embeddings = torch.randn(batch_size, 1280, device=device)
    
    # Compute rewards
    print(f"\n📊 Computing rewards for {len(molecule_graphs)} molecules...")
    reward_dict = reward_function.compute_reward(
        molecule_graphs=molecule_graphs,
        protein_embeddings=protein_embeddings,
        smiles_list=valid_smiles
    )
    
    # Display results
    print("\n🎯 Reward Results:")
    print("-" * 40)
    for i, smiles in enumerate(valid_smiles):
        print(f"Molecule {i+1}: {smiles}")
        print(f"  DTI Score:    {reward_dict['dti'][i]:.3f}")
        print(f"  ADMET Score:  {reward_dict['admet'][i]:.3f}")
        print(f"  Total Reward: {reward_dict['total'][i]:.3f}")
        print()
    
    print("📈 Batch Statistics:")
    print(f"  Mean DTI:    {reward_dict['dti'].mean():.3f} ± {reward_dict['dti'].std():.3f}")
    print(f"  Mean ADMET:  {reward_dict['admet'].mean():.3f} ± {reward_dict['admet'].std():.3f}")
    print(f"  Mean Total:  {reward_dict['total'].mean():.3f} ± {reward_dict['total'].std():.3f}")
    
    # Test normalization stats
    print(f"\n📊 Normalization Stats:")
    stats = reward_function.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n✅ Reward Function test complete!")

def test_sample_for_rl():
    """Test the sample_for_rl method."""
    print("🧪 Testing sample_for_rl Method")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import the diffusion model
    from models.graph_diffusion import ConditionalGraphDiffusion
    
    # Create a small model for testing
    model = ConditionalGraphDiffusion(
        atom_feature_dim=7,
        edge_feature_dim=3,
        protein_dim=1280,
        hidden_dim=128,  # Smaller for testing
        num_layers=4,    # Fewer layers for testing
        max_atoms=20,    # Smaller molecules
        timesteps=100    # Fewer timesteps for testing
    ).to(device)
    
    # Test parameters
    batch_size = 2
    num_samples_per_protein = 4
    protein_embeddings = torch.randn(batch_size, 1280, device=device)
    
    print(f"Generating {num_samples_per_protein} molecules per protein...")
    print(f"Batch size: {batch_size}, Total samples: {batch_size * num_samples_per_protein}")
    
    # Test sampling with gradients
    molecules, log_probs = model.sample_for_rl(
        protein_embeddings=protein_embeddings,
        num_samples_per_protein=num_samples_per_protein,
        temperature=1.0,
        return_log_probs=True
    )
    
    print(f"\n📊 Generation Results:")
    print(f"  Generated molecules: {len(molecules.to_data_list())}")
    print(f"  Log probabilities shape: {log_probs.shape}")
    print(f"  Log probabilities require grad: {log_probs.requires_grad}")
    
    # Convert some to SMILES
    from models.graph_diffusion import graph_to_smiles
    
    print(f"\n🧬 Sample Molecules:")
    for i, mol_data in enumerate(molecules.to_data_list()[:5]):  # Show first 5
        smiles = graph_to_smiles(mol_data)
        atoms = mol_data.x.shape[0]
        edges = mol_data.edge_index.shape[1] // 2  # Undirected edges
        print(f"  Molecule {i+1}: {smiles}")
        print(f"    Atoms: {atoms}, Edges: {edges}")
        print(f"    Log prob: {log_probs[i]:.3f}")
    
    # Test REINFORCE update
    print(f"\n🔄 Testing REINFORCE Update:")
    fake_rewards = torch.randn(len(log_probs), device=device)
    loss = model.reinforce_update(fake_rewards, log_probs)
    
    print(f"  Fake rewards shape: {fake_rewards.shape}")
    print(f"  REINFORCE loss: {loss.item():.3f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    
    print("\n✅ sample_for_rl test complete!")

def main():
    """Run all tests."""
    print("🧪 REWARD FUNCTION & RL TESTING")
    print("=" * 60)
    
    try:
        test_admet_scorer()
        test_reward_function_mock()
        test_sample_for_rl()
        
        print("🎉 ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
