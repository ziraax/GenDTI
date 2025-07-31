#!/usr/bin/env python3
"""
Test protein-specific molecular generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.graph_diffusion import ConditionalGraphDiffusion, graph_to_smiles
from models.esm_encoder import ESMEncoder

def test_protein_specific_generation():
    """Test if different proteins generate different molecules."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print("📦 Loading model...")
    with open('configs/diffusion_conditional.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    model = ConditionalGraphDiffusion(**model_config)
    
    checkpoint = torch.load('outputs/stage2_conditional_v4/best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load protein encoder
    protein_encoder = ESMEncoder(model_name='facebook/esm2_t6_8M_UR50D')
    
    # Test with different proteins
    test_proteins = [
        ("Short protein", "MKTAYIAKQRQISFVKSHFSRQLEERLG"),  # 28 amino acids
        ("Medium protein", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"),  # 140 amino acids  
        ("Hydrophobic protein", "LLLLLLLLLLLLLLLLLLLLLLLLLLL"),  # Very hydrophobic
        ("Polar protein", "KKKKKKKNNNNNNNSSSSSSSEEEEEEE")   # Very polar
    ]
    
    all_results = {}
    
    for protein_name, protein_seq in test_proteins:
        print(f"\n🧬 Testing {protein_name} (length: {len(protein_seq)})")
        
        # Encode protein
        protein_emb = protein_encoder(protein_seq)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)
        
        # Generate molecules
        with torch.no_grad():
            generated_batch = model.sample_realistic(
                protein_emb.repeat(3, 1),  # Generate 3 molecules
                num_atoms=None,
                edge_threshold=0.3,
                max_degree=4
            )
            
            generated_graphs = generated_batch.to_data_list()
        
        # Convert to SMILES
        smiles_results = []
        for i, graph in enumerate(generated_graphs):
            smiles = graph_to_smiles(graph, sanitize=True)
            num_atoms = graph.x.shape[0]
            if smiles:
                smiles_results.append(smiles)
                print(f"  Molecule {i+1}: {smiles} ({num_atoms} atoms)")
        
        all_results[protein_name] = {
            'length': len(protein_seq),
            'smiles': smiles_results
        }
    
    # Analyze diversity
    print(f"\n{'='*60}")
    print("PROTEIN-SPECIFIC GENERATION ANALYSIS")
    print(f"{'='*60}")
    
    all_smiles = set()
    for protein_name, data in all_results.items():
        unique_smiles = set(data['smiles'])
        all_smiles.update(unique_smiles)
        
        print(f"\n{protein_name}:")
        print(f"  Unique molecules: {len(unique_smiles)}/{len(data['smiles'])}")
        for smiles in unique_smiles:
            print(f"    • {smiles}")
    
    print(f"\nTotal unique molecules across all proteins: {len(all_smiles)}")
    print(f"Diversity score: {len(all_smiles)/(4*3)*100:.1f}% (higher is better)")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (protein_name, data) in enumerate(all_results.items()):
        ax = axes[i]
        
        # Display unique molecules for this protein
        unique_smiles = list(set(data['smiles']))
        text = f"{protein_name}\n(Length: {data['length']})\n\n"
        
        for j, smiles in enumerate(unique_smiles):
            text += f"{j+1}. {smiles}\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               fontsize=8, fontfamily='monospace', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/protein_specific_generation.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Protein comparison saved to: outputs/protein_specific_generation.png")
    
    return all_results

if __name__ == "__main__":
    try:
        results = test_protein_specific_generation()
        print(f"\n🎉 Success! Tested protein-specific generation.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
