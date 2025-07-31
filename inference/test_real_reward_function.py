"""
Test the complete RL reward function with actual DTI and GNN models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.reward import RewardFunction
from models.dti_model import DTIModel
from models.gnn_encoder import GNNEncoder
from models.graph_diffusion import create_molecule_graph


def load_dti_model(model_path: str, device: torch.device) -> DTIModel:
    """Load the trained DTI model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']['model']
    
    # Your DTI model configuration from the checkpoint
    model = DTIModel(
        drug_dim=256,  # From GNN encoder output
        protein_dim=320,  # From your protein embeddings
        fusion=config['fusion'],
        proj_dim=config['proj_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_gnn_encoder(model_path: str, device: torch.device) -> GNNEncoder:
    """Load the GNN encoder model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = GNNEncoder(
        input_dim=config['input_dim'],
        edge_dim=config['edge_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def test_reward_function():
    """Test the complete reward function with real models."""
    print("🧪 Testing Complete RL Reward Function")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading DTI model...")
    dti_model = load_dti_model(
        'outputs/production_final_5/best_model.pt', 
        device
    )
    
    print("Loading GNN encoder...")
    gnn_encoder = load_gnn_encoder(
        'outputs/drug_embeddings/drug_embeddings_gnn_model.pt',
        device
    )
    
    # Initialize reward function
    print("Initializing reward function...")
    reward_function = RewardFunction(
        dti_model=dti_model,
        drug_encoder=gnn_encoder,
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
    
    print(f"Testing {len(test_smiles)} molecules...")
    
    # Convert SMILES to graphs
    molecule_graphs = []
    valid_smiles = []
    
    for smiles in test_smiles:
        graph = create_molecule_graph(smiles)
        if graph is not None:
            molecule_graphs.append(graph)
            valid_smiles.append(smiles)
            print(f"  ✅ Converted: {smiles}")
        else:
            print(f"  ❌ Failed: {smiles}")
    
    if len(molecule_graphs) == 0:
        print("❌ No valid molecules to test!")
        return
    
    # Create mock protein embeddings (320D to match your protein embeddings)
    batch_size = len(molecule_graphs)
    protein_embeddings = torch.randn(batch_size, 320, device=device)
    
    print(f"\n🎯 Computing rewards...")
    
    # Test DTI scoring
    print("Testing DTI scoring...")
    try:
        dti_scores = reward_function.compute_dti_scores(molecule_graphs, protein_embeddings)
        print(f"  ✅ DTI scores computed: {dti_scores}")
    except Exception as e:
        print(f"  ❌ DTI scoring failed: {e}")
        return
    
    # Test ADMET scoring
    print("Testing ADMET scoring...")
    try:
        admet_scores = reward_function.compute_admet_scores(valid_smiles)
        print(f"  ✅ ADMET scores computed: {admet_scores}")
    except Exception as e:
        print(f"  ❌ ADMET scoring failed: {e}")
        return
    
    # Test combined reward
    print("Testing combined reward computation...")
    try:
        reward_dict = reward_function.compute_reward(
            molecule_graphs=molecule_graphs,
            protein_embeddings=protein_embeddings,
            smiles_list=valid_smiles
        )
        
        print(f"\n📊 Results:")
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
        
        print("\n🎉 All tests passed! Reward function is working correctly.")
        
    except Exception as e:
        print(f"  ❌ Combined reward failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_reward_function()
