#!/usr/bin/env python3
"""
Molecular visualization pipeline: Model Output → SMILES → Visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import json
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.graph_diffusion import ConditionalGraphDiffusion, graph_to_smiles
from models.esm_encoder import ESMEncoder
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_test_proteins():
    """Load test protein sequences from data."""
    try:
        # Load test data
        test_data = pd.read_csv('data/processed/test.tsv', sep='\t')
        
        # Get unique proteins (sample a few for testing)
        unique_proteins = test_data['BindingDB_Target_Chain_Sequence'].unique()
        
        # Select diverse test proteins (different lengths)
        protein_lengths = [(seq, len(seq)) for seq in unique_proteins[:50]]
        protein_lengths.sort(key=lambda x: x[1])
        
        # Select proteins of different sizes
        test_proteins = []
        for i in range(0, len(protein_lengths), len(protein_lengths)//5):
            if i < len(protein_lengths):
                test_proteins.append(protein_lengths[i][0])
        
        # Add some specific interesting proteins if available
        test_proteins = test_proteins[:5]  # Limit to 5 for testing
        
        print(f"✓ Loaded {len(test_proteins)} test proteins")
        return test_proteins
                
    except Exception as e:
        print(f"✗ Error loading test proteins: {e}")
        # Fallback to dummy proteins
        return [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFKLI",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
            "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
        ]

class MolecularVisualizer:
    """Visualizes generated molecules from model output."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model config (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, config_path)
        self.protein_encoder = ESMEncoder(model_name='facebook/esm2_t6_8M_UR50D')
        
        # Test proteins
        self.test_proteins = get_test_proteins()
        
        print(f"✅ Model loaded successfully")
        print(f"🧬 {len(self.test_proteins)} test proteins available")
    
    def _load_model(self, model_path: str, config_path: str = None) -> ConditionalGraphDiffusion:
        """Load the trained model."""
        try:
            import yaml
            
            # Load config
            if config_path is None:
                config_path = 'configs/diffusion_conditional.yaml'
                
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                
            model_config = full_config['model']
            
            print(f"📋 Using model config:")
            for k, v in model_config.items():
                print(f"   {k}: {v}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model with correct config
            model = ConditionalGraphDiffusion(**model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'loss' in checkpoint:
                    print(f"📊 Model training loss: {checkpoint['loss']:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Loaded model state dict directly")
                
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_molecules(self, protein_idx: int = 0, num_molecules: int = 5, 
                          edge_threshold: float = 0.3, max_degree: int = 4) -> List[Dict[str, Any]]:
        """
        Generate molecules for a specific protein.
        
        Args:
            protein_idx: Index of protein to use
            num_molecules: Number of molecules to generate
            edge_threshold: Edge probability threshold
            max_degree: Maximum degree per atom
            
        Returns:
            List of molecule data with graphs and SMILES
        """
        if protein_idx >= len(self.test_proteins):
            raise ValueError(f"Protein index {protein_idx} out of range (0-{len(self.test_proteins)-1})")
        
        protein_seq = self.test_proteins[protein_idx]
        print(f"🧬 Generating {num_molecules} molecules for protein {protein_idx+1} (length: {len(protein_seq)})")
        
        # Encode protein
        protein_emb = self.protein_encoder(protein_seq)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)
        
        # Generate molecules
        try:
            with torch.no_grad():
                generated_batch = self.model.sample_realistic(
                    protein_emb.repeat(num_molecules, 1),
                    num_atoms=None,  # Variable sizes
                    edge_threshold=edge_threshold,
                    max_degree=max_degree
                )
                
                generated_graphs = generated_batch.to_data_list()
                
            molecules = []
            print(f"📊 Converting {len(generated_graphs)} graphs to SMILES...")
            
            for i, graph in enumerate(generated_graphs):
                try:
                    # Convert to SMILES
                    smiles = graph_to_smiles(graph, sanitize=True)
                    
                    # Calculate basic properties
                    num_atoms = graph.x.shape[0]
                    num_edges = graph.edge_index.shape[1]
                    avg_degree = (2 * num_edges) / num_atoms if num_atoms > 0 else 0
                    
                    molecule_data = {
                        'molecule_idx': i,
                        'smiles': smiles,
                        'graph': graph,
                        'num_atoms': num_atoms,
                        'num_edges': num_edges,
                        'avg_degree': avg_degree,
                        'protein_idx': protein_idx,
                        'protein_length': len(protein_seq),
                        'generation_successful': smiles is not None
                    }
                    
                    molecules.append(molecule_data)
                    
                    if smiles:
                        print(f"  ✅ Molecule {i+1}: {smiles} ({num_atoms} atoms, {num_edges} edges)")
                    else:
                        print(f"  ❌ Molecule {i+1}: SMILES conversion failed")
                        
                except Exception as e:
                    print(f"  ❌ Error processing molecule {i+1}: {e}")
                    molecules.append({
                        'molecule_idx': i,
                        'smiles': None,
                        'error': str(e),
                        'generation_successful': False
                    })
            
            successful = sum(1 for m in molecules if m['generation_successful'])
            print(f"✅ Generated {successful}/{len(molecules)} valid SMILES")
            
            return molecules
            
        except Exception as e:
            print(f"❌ Error in molecule generation: {e}")
            return []
    
    def visualize_smiles_2d(self, smiles: str, title: str = None, ax=None) -> bool:
        """
        Visualize a SMILES string as 2D molecular structure.
        
        Args:
            smiles: SMILES string
            title: Plot title
            ax: Matplotlib axis (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try RDKit visualization first
            return self._visualize_with_rdkit(smiles, title, ax)
        except ImportError:
            print("⚠️  RDKit not available. Using text representation.")
            self._visualize_smiles_text(smiles, title, ax)
            return True
        except Exception as e:
            print(f"❌ Error with RDKit visualization: {e}")
            self._visualize_smiles_text(smiles, title, ax)
            return True
    
    def _visualize_with_rdkit(self, smiles: str, title: str = None, ax=None) -> bool:
        """Try to visualize with RDKit."""
        from rdkit import Chem
        from rdkit.Chem import Draw
        import matplotlib.pyplot as plt
        import io
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Create 2D representation using matplotlib directly
        img = Draw.MolToImage(mol, size=(400, 400))
        
        # Plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.imshow(img)
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=10, pad=10)
        
        return True
    
    def _visualize_smiles_text(self, smiles: str, title: str = None, ax=None):
        """Fallback text visualization when RDKit is not available."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        # Create a nice text box
        bbox = FancyBboxPatch((0.05, 0.2), 0.9, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor='lightblue', 
                             edgecolor='navy', 
                             linewidth=1)
        ax.add_patch(bbox)
        
        # Add SMILES text with word wrapping
        smiles_display = smiles
        if len(smiles) > 40:
            # Split long SMILES for better display
            mid = len(smiles) // 2
            smiles_display = smiles[:mid] + '\n' + smiles[mid:]
        
        ax.text(0.5, 0.5, smiles_display, ha='center', va='center', 
               transform=ax.transAxes, fontsize=8, fontfamily='monospace',
               wrap=True)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=10, pad=10)
    
    def create_molecule_gallery(self, molecules: List[Dict[str, Any]], 
                              output_path: str = None, max_display: int = 12) -> str:
        """
        Create a gallery visualization of generated molecules.
        
        Args:
            molecules: List of molecule data
            output_path: Output file path (optional)
            max_display: Maximum number of molecules to display
            
        Returns:
            Path to saved figure
        """
        successful_molecules = [m for m in molecules if m['generation_successful'] and m['smiles']]
        
        if not successful_molecules:
            print("❌ No valid molecules to visualize")
            return None
        
        n_molecules = min(len(successful_molecules), max_display)
        successful_molecules = successful_molecules[:n_molecules]
        
        # Calculate grid layout
        cols = min(4, n_molecules)
        rows = (n_molecules + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Handle different subplot configurations
        if n_molecules == 1:
            axes = [axes]  # Single subplot
        elif rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]  # Single row
        else:
            axes = axes.flatten()  # Multiple rows - flatten to 1D array
        
        # Plot each molecule
        for i, mol_data in enumerate(successful_molecules):
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) and len(axes) > 1 else axes[0] if isinstance(axes, (list, np.ndarray)) else axes
            
            smiles = mol_data['smiles']
            title = (f"Molecule {mol_data['molecule_idx']+1}\n"
                    f"{mol_data['num_atoms']} atoms, {mol_data['num_edges']} edges\n"
                    f"Avg degree: {mol_data['avg_degree']:.1f}")
            
            self.visualize_smiles_2d(smiles, title, ax)
        
        # Hide unused subplots
        if isinstance(axes, (list, np.ndarray)) and len(axes) > n_molecules:
            for i in range(n_molecules, len(axes)):
                axes[i].axis('off')
        
        # Add main title
        protein_idx = successful_molecules[0]['protein_idx']
        protein_len = successful_molecules[0]['protein_length']
        fig.suptitle(f"Generated Molecules for Protein {protein_idx+1} (length: {protein_len})", 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/visualizations/molecule_gallery_protein{protein_idx+1}_{timestamp}.png"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Gallery saved to: {output_path}")
        return output_path
    
    def analyze_and_visualize(self, protein_idx: int = 0, num_molecules: int = 8,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: generate, analyze, and visualize molecules.
        
        Args:
            protein_idx: Protein to use
            num_molecules: Number of molecules to generate
            save_results: Whether to save detailed results
            
        Returns:
            Complete analysis results
        """
        print(f"\n{'='*60}")
        print("MOLECULAR GENERATION & VISUALIZATION PIPELINE")
        print(f"{'='*60}")
        
        # Generate molecules
        molecules = self.generate_molecules(protein_idx, num_molecules)
        
        if not molecules:
            print("❌ No molecules generated")
            return {}
        
        # Create visualization
        gallery_path = self.create_molecule_gallery(molecules)
        
        # Analyze results
        successful = [m for m in molecules if m['generation_successful']]
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'protein_idx': protein_idx,
            'protein_length': molecules[0]['protein_length'] if molecules else 0,
            'total_generated': len(molecules),
            'successful_conversions': len(successful),
            'success_rate': len(successful) / len(molecules) if molecules else 0,
            'molecules': molecules,
            'gallery_path': gallery_path
        }
        
        if successful:
            smiles_list = [m['smiles'] for m in successful]
            atom_counts = [m['num_atoms'] for m in successful]
            avg_degrees = [m['avg_degree'] for m in successful]
            
            analysis['statistics'] = {
                'unique_smiles': len(set(smiles_list)),
                'avg_atoms': np.mean(atom_counts),
                'std_atoms': np.std(atom_counts),
                'avg_degree_mean': np.mean(avg_degrees),
                'avg_degree_std': np.std(avg_degrees),
                'smiles_list': smiles_list
            }
            
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"   Success rate: {analysis['success_rate']*100:.1f}%")
            print(f"   Unique SMILES: {analysis['statistics']['unique_smiles']}/{len(successful)}")
            print(f"   Molecule sizes: {analysis['statistics']['avg_atoms']:.1f} ± {analysis['statistics']['std_atoms']:.1f} atoms")
            print(f"   Average degree: {analysis['statistics']['avg_degree_mean']:.2f} ± {analysis['statistics']['avg_degree_std']:.2f}")
            
            print(f"\n🧬 GENERATED SMILES:")
            for i, smiles in enumerate(smiles_list):
                print(f"   {i+1:2d}. {smiles}")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"outputs/visualizations/molecule_analysis_protein{protein_idx+1}_{timestamp}.json"
            
            # Make JSON serializable
            json_analysis = self._make_json_serializable(analysis)
            
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(json_analysis, f, indent=2)
            
            print(f"💾 Detailed results saved to: {results_path}")
        
        return analysis
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items() 
                   if key != 'graph'}  # Skip PyTorch graphs
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize generated molecules')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--protein', type=int, default=0, help='Protein index to use (0-based)')
    parser.add_argument('--num_molecules', type=int, default=8, help='Number of molecules to generate')
    parser.add_argument('--edge_threshold', type=float, default=0.3, help='Edge probability threshold')
    parser.add_argument('--max_degree', type=int, default=4, help='Maximum degree per atom')
    parser.add_argument('--output_dir', default='outputs/visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Create visualizer
    try:
        visualizer = MolecularVisualizer(args.model)
        
        # Run analysis
        results = visualizer.analyze_and_visualize(
            protein_idx=args.protein,
            num_molecules=args.num_molecules,
            save_results=True
        )
        
        if results:
            print(f"\n✅ Visualization complete!")
            print(f"📊 Generated {results['successful_conversions']}/{results['total_generated']} valid molecules")
            if 'gallery_path' in results and results['gallery_path']:
                print(f"🖼️  Gallery: {results['gallery_path']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
