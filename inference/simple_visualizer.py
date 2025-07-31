#!/usr/bin/env python3
"""
Simple molecular generation and SMILES visualization
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

def simple_molecule_generation():
    """Simple test of molecule generation and SMILES conversion."""
    
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
    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load protein encoder
    print("🧬 Loading protein encoder...")
    protein_encoder = ESMEncoder(model_name='facebook/esm2_t6_8M_UR50D')
    print("✅ Protein encoder loaded")
    
    # Test protein (simple short sequence)
    test_protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFKLI"
    print(f"🧬 Test protein length: {len(test_protein)}")
    
    # Encode protein
    print("🔄 Encoding protein...")
    protein_emb = protein_encoder(test_protein)
    if protein_emb.dim() == 1:
        protein_emb = protein_emb.unsqueeze(0)
    print(f"✅ Protein encoded: {protein_emb.shape}")
    
    # Generate molecules
    print("🎲 Generating molecules...")
    num_molecules = 3
    
    with torch.no_grad():
        generated_batch = model.sample_realistic(
            protein_emb.repeat(num_molecules, 1),
            num_atoms=None,
            edge_threshold=0.3,
            max_degree=4
        )
        
        generated_graphs = generated_batch.to_data_list()
    
    print(f"✅ Generated {len(generated_graphs)} molecular graphs")
    
    # Convert to SMILES
    print("📊 Converting to SMILES...")
    smiles_results = []
    
    for i, graph in enumerate(generated_graphs):
        smiles = graph_to_smiles(graph, sanitize=True)
        num_atoms = graph.x.shape[0]
        num_edges = graph.edge_index.shape[1]
        
        print(f"  Molecule {i+1}: {num_atoms} atoms, {num_edges} edges")
        if smiles:
            print(f"    SMILES: {smiles}")
            smiles_results.append(smiles)
        else:
            print(f"    ❌ SMILES conversion failed")
    
    print(f"\n✅ Successfully generated {len(smiles_results)}/{len(generated_graphs)} SMILES")
    
    # Simple visualization with matplotlib
    if smiles_results:
        print("🖼️  Creating simple visualization...")
        
        fig, axes = plt.subplots(1, len(smiles_results), figsize=(5*len(smiles_results), 4))
        if len(smiles_results) == 1:
            axes = [axes]
        
        for i, smiles in enumerate(smiles_results):
            ax = axes[i]
            
            # Simple text display
            ax.text(0.5, 0.5, smiles, ha='center', va='center', 
                   fontsize=10, fontfamily='monospace', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
            
            ax.set_title(f"Molecule {i+1}", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = "outputs/simple_molecule_visualization.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Visualization saved to: {output_path}")
        
        # Also try RDKit if available
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            print("🧪 Creating RDKit visualization...")
            mols = []
            for smiles in smiles_results:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
            
            if mols:
                img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300))
                rdkit_path = "outputs/rdkit_molecule_visualization.png"
                img.save(rdkit_path)
                print(f"💾 RDKit visualization saved to: {rdkit_path}")
            
        except ImportError:
            print("⚠️  RDKit not available for advanced visualization")
        except Exception as e:
            print(f"⚠️  RDKit visualization error: {e}")
    
    return smiles_results

if __name__ == "__main__":
    try:
        results = simple_molecule_generation()
        print(f"\n🎉 Success! Generated {len(results)} valid molecules.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
