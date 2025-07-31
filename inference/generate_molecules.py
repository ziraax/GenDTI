"""
Molecular generation and evaluation utilities.
"""

import torch
import numpy as np
from torch_geometric.data import Batch
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.graph_diffusion import ConditionalGraphDiffusion
from models.diffusion_utils import GraphDiffusionProcess, DiffusionSchedule

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Some evaluation features will be limited.")
    RDKIT_AVAILABLE = False


class MolecularGenerator:
    """
    Molecular generator using trained diffusion model.
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        self.model = ConditionalGraphDiffusion(**config['model']).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup diffusion process
        schedule = DiffusionSchedule(**config.get('diffusion', {}))
        self.diffusion = GraphDiffusionProcess(schedule)
        
        print(f"Loaded model from {model_path}")
    
    @torch.no_grad()
    def generate(
        self,
        protein_embeddings: torch.Tensor,
        num_molecules: int = 1,
        num_atoms: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate molecules conditioned on protein embeddings.
        
        Args:
            protein_embeddings: Protein conditioning [batch_size, protein_dim]
            num_molecules: Number of molecules to generate per protein
            num_atoms: Number of atoms per molecule [batch_size]
            return_intermediates: Whether to return generation intermediates
            
        Returns:
            List of generated molecules with metadata
        """
        batch_size = protein_embeddings.size(0)
        
        # Expand protein embeddings if generating multiple molecules per protein
        if num_molecules > 1:
            protein_embeddings = protein_embeddings.repeat_interleave(num_molecules, dim=0)
            if num_atoms is not None:
                num_atoms = num_atoms.repeat_interleave(num_molecules)
        
        # Generate molecules
        generated_graphs = self.diffusion.p_sample_loop(
            self.model,
            protein_embeddings,
            num_atoms=num_atoms,
            device=self.device
        )
        
        # Convert to list of molecules
        molecules = []
        batch_ptr = generated_graphs.ptr if hasattr(generated_graphs, 'ptr') else None
        
        for i in range(protein_embeddings.size(0)):
            if batch_ptr is not None:
                start_idx = batch_ptr[i].item()
                end_idx = batch_ptr[i + 1].item()
            else:
                # Fallback if ptr not available
                molecule_mask = generated_graphs.batch == i
                start_idx = torch.where(molecule_mask)[0][0].item()
                end_idx = torch.where(molecule_mask)[0][-1].item() + 1
            
            # Extract molecule data
            mol_data = {
                'atoms': generated_graphs.x[start_idx:end_idx],
                'coordinates': generated_graphs.pos[start_idx:end_idx],
                'edge_index': generated_graphs.edge_index[:, 
                    (generated_graphs.edge_index[0] >= start_idx) & 
                    (generated_graphs.edge_index[0] < end_idx)
                ] - start_idx,
                'edge_attr': generated_graphs.edge_attr[
                    (generated_graphs.edge_index[0] >= start_idx) & 
                    (generated_graphs.edge_index[0] < end_idx)
                ] if hasattr(generated_graphs, 'edge_attr') else None,
                'protein_idx': i // num_molecules,
                'molecule_idx': i % num_molecules
            }
            
            molecules.append(mol_data)
        
        return molecules
    
    def molecules_to_smiles(self, molecules: List[Dict[str, Any]]) -> List[str]:
        """
        Convert generated molecules to SMILES strings.
        This is a placeholder - you'll need to implement graph->SMILES conversion.
        """
        if not RDKIT_AVAILABLE:
            return ["CCO" for _ in molecules]  # Dummy SMILES
        
        smiles_list = []
        for mol_data in molecules:
            try:
                # TODO: Implement proper graph to SMILES conversion
                # This would involve:
                # 1. Converting atom indices back to element symbols
                # 2. Reconstructing bonds from edge information
                # 3. Creating RDKit molecule object
                # 4. Converting to SMILES
                
                smiles = "CCO"  # Placeholder
                smiles_list.append(smiles)
            except Exception as e:
                print(f"Error converting molecule to SMILES: {e}")
                smiles_list.append(None)
        
        return smiles_list


class MolecularEvaluator:
    """
    Evaluator for generated molecules.
    Computes various molecular properties and drug-likeness metrics.
    """
    
    def __init__(self):
        self.metrics = [
            'validity',
            'uniqueness', 
            'novelty',
            'drug_likeness',
            'diversity',
            'fcd_score'  # Fréchet ChemNet Distance
        ]
    
    def evaluate_molecules(
        self,
        generated_smiles: List[str],
        reference_smiles: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate generated molecules against various metrics.
        
        Args:
            generated_smiles: List of generated SMILES
            reference_smiles: Reference SMILES for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Filter valid SMILES
        valid_smiles = [s for s in generated_smiles if s is not None and self._is_valid_smiles(s)]
        
        # Basic metrics
        results['validity'] = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0
        results['uniqueness'] = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0
        
        if reference_smiles:
            reference_set = set(reference_smiles)
            novel_molecules = [s for s in valid_smiles if s not in reference_set]
            results['novelty'] = len(novel_molecules) / len(valid_smiles) if valid_smiles else 0
        
        # Drug-likeness
        if RDKIT_AVAILABLE and valid_smiles:
            drug_like_scores = []
            for smiles in valid_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Lipinski's Rule of Five
                        mw = Descriptors.MolWt(mol)
                        logp = Crippen.MolLogP(mol)
                        hbd = Descriptors.NumHDonors(mol)
                        hba = Descriptors.NumHAcceptors(mol)
                        
                        # Check Lipinski's rules
                        drug_like = (
                            mw <= 500 and
                            logp <= 5 and
                            hbd <= 5 and
                            hba <= 10
                        )
                        drug_like_scores.append(float(drug_like))
                except:
                    continue
            
            results['drug_likeness'] = np.mean(drug_like_scores) if drug_like_scores else 0
        
        # Diversity (average pairwise Tanimoto distance)
        if RDKIT_AVAILABLE and len(valid_smiles) > 1:
            results['diversity'] = self._calculate_diversity(valid_smiles)
        
        return results
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid."""
        if not RDKIT_AVAILABLE:
            return smiles is not None and len(smiles) > 0
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _calculate_diversity(self, smiles_list: List[str]) -> float:
        """Calculate average pairwise Tanimoto diversity."""
        if not RDKIT_AVAILABLE:
            return 0.5  # Dummy value
        
        try:
            from rdkit.Chem import rdMolDescriptors
            from rdkit import DataStructs
            
            # Calculate fingerprints
            fps = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            
            if len(fps) < 2:
                return 0.0
            
            # Calculate pairwise Tanimoto similarities
            similarities = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
            
            # Diversity is 1 - average similarity
            return 1.0 - np.mean(similarities)
            
        except Exception as e:
            print(f"Error calculating diversity: {e}")
            return 0.0
    
    def molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties for a single molecule."""
        if not RDKIT_AVAILABLE:
            return {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            props = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'qed_score': QED.qed(mol)  # Quantitative Estimate of Drug-likeness
            }
            
            return props
            
        except Exception as e:
            print(f"Error calculating properties for {smiles}: {e}")
            return {}


def load_generator(model_path: str, config_path: str) -> MolecularGenerator:
    """Load trained diffusion model for generation."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return MolecularGenerator(model_path, config)


def generate_molecules_for_proteins(
    generator: MolecularGenerator,
    protein_embeddings: torch.Tensor,
    num_molecules_per_protein: int = 10,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate molecules for given protein embeddings.
    
    Args:
        generator: Trained molecular generator
        protein_embeddings: Protein embeddings [N, protein_dim]
        num_molecules_per_protein: Number of molecules to generate per protein
        output_file: Optional file to save results
        
    Returns:
        List of generation results
    """
    print(f"Generating {num_molecules_per_protein} molecules for {protein_embeddings.size(0)} proteins...")
    
    # Generate molecules
    molecules = generator.generate(
        protein_embeddings,
        num_molecules=num_molecules_per_protein
    )
    
    # Convert to SMILES
    smiles_list = generator.molecules_to_smiles(molecules)
    
    # Evaluate
    evaluator = MolecularEvaluator()
    metrics = evaluator.evaluate_molecules(smiles_list)
    
    # Organize results
    results = []
    for i, (mol_data, smiles) in enumerate(zip(molecules, smiles_list)):
        result = {
            'protein_idx': mol_data['protein_idx'],
            'molecule_idx': mol_data['molecule_idx'],
            'smiles': smiles,
            'valid': smiles is not None and evaluator._is_valid_smiles(smiles),
            'properties': evaluator.molecular_properties(smiles) if smiles else {}
        }
        results.append(result)
    
    # Add global metrics
    generation_summary = {
        'total_molecules': len(results),
        'num_proteins': protein_embeddings.size(0),
        'molecules_per_protein': num_molecules_per_protein,
        'metrics': metrics
    }
    
    # Save results if requested
    if output_file:
        import json
        output_data = {
            'generation_summary': generation_summary,
            'molecules': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    print("Generation Summary:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return results
