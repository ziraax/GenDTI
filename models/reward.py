"""
Reward function module for RL fine-tuning of molecular generation.
Combines DTI binding affinity and ADMET properties.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from torch_geometric.data import Data, Batch

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. ADMET scoring will be disabled.")


class ADMETScorer:
    """
    ADMET property calculator using RDKit descriptors.
    Computes various drug-likeness and ADMET-related properties.
    """
    
    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ADMET scoring")
            
        # Initialize PAINS filter for toxic substructures
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_filter = FilterCatalog(params)
        except:
            self.pains_filter = None
            print("Warning: PAINS filter not available")
    
    def compute_lipinski_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Compute Lipinski Rule of 5 properties."""
        if mol is None:
            return {'mw': 0.0, 'logp': 0.0, 'hbd': 0.0, 'hba': 0.0}
            
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            return {
                'mw': float(mw),
                'logp': float(logp), 
                'hbd': float(hbd),
                'hba': float(hba)
            }
        except:
            return {'mw': 0.0, 'logp': 0.0, 'hbd': 0.0, 'hba': 0.0}
    
    def compute_admet_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Compute extended ADMET-relevant properties."""
        if mol is None:
            return self._default_admet_dict()
            
        try:
            # Basic descriptors
            tpsa = Descriptors.TPSA(mol)  # Topological polar surface area
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            heavy_atoms = Descriptors.HeavyAtomCount(mol)
            
            # Drug-likeness indicators
            qed = self._compute_qed(mol)  # Quantitative drug-likeness
            
            # Synthetic accessibility (approximation)
            sa_score = self._compute_sa_score(mol)
            
            # PAINS alerts (toxic substructures)
            pains_alerts = self._compute_pains_alerts(mol)
            
            return {
                'tpsa': float(tpsa),
                'rotatable_bonds': float(rotatable_bonds),
                'aromatic_rings': float(aromatic_rings),
                'heavy_atoms': float(heavy_atoms),
                'qed': float(qed),
                'sa_score': float(sa_score),
                'pains_alerts': float(pains_alerts)
            }
        except Exception as e:
            print(f"Error computing ADMET properties: {e}")
            return self._default_admet_dict()
    
    def _compute_qed(self, mol: Chem.Mol) -> float:
        """Compute QED (Quantitative Drug-likeness) score."""
        try:
            from rdkit.Chem.QED import qed
            return qed(mol)
        except:
            # Fallback: simple approximation based on Lipinski-like rules
            props = self.compute_lipinski_properties(mol)
            
            # Simple drug-likeness heuristic (0-1 scale)
            mw_score = 1.0 if 150 <= props['mw'] <= 500 else max(0.0, 1.0 - abs(props['mw'] - 325) / 325)
            logp_score = 1.0 if -0.4 <= props['logp'] <= 5.6 else max(0.0, 1.0 - abs(props['logp'] - 2.6) / 3.0)
            hbd_score = 1.0 if props['hbd'] <= 5 else max(0.0, 1.0 - (props['hbd'] - 5) / 5.0)
            hba_score = 1.0 if props['hba'] <= 10 else max(0.0, 1.0 - (props['hba'] - 10) / 10.0)
            
            return (mw_score + logp_score + hbd_score + hba_score) / 4.0
    
    def _compute_sa_score(self, mol: Chem.Mol) -> float:
        """Approximate synthetic accessibility score (higher = more accessible)."""
        try:
            # Simple heuristic based on molecular complexity
            heavy_atoms = mol.GetNumHeavyAtoms()
            rings = rdMolDescriptors.CalcNumRings(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Normalize complexity (lower complexity = higher SA score)
            complexity = heavy_atoms + rings * 2 + rotatable_bonds * 0.5
            sa_score = max(0.0, 1.0 - complexity / 50.0)  # Scale to [0,1]
            
            return sa_score
        except:
            return 0.5  # Neutral score
    
    def _compute_pains_alerts(self, mol: Chem.Mol) -> float:
        """Count PAINS alerts (lower is better)."""
        if self.pains_filter is None:
            return 0.0
            
        try:
            matches = self.pains_filter.GetMatches(mol)
            # Convert to a score where 0 = no alerts (good), 1 = many alerts (bad)
            return min(1.0, len(matches) / 3.0)
        except:
            return 0.0
    
    def _default_admet_dict(self) -> Dict[str, float]:
        """Return default ADMET properties for failed molecules."""
        return {
            'tpsa': 0.0,
            'rotatable_bonds': 0.0,
            'aromatic_rings': 0.0,
            'heavy_atoms': 0.0,
            'qed': 0.0,
            'sa_score': 0.0,
            'pains_alerts': 1.0  # Assume worst case for failed molecules
        }
    
    def score_molecule(self, smiles: str) -> float:
        """
        Compute overall ADMET score for a molecule (0-1 scale, higher is better).
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            ADMET score between 0 and 1
        """
        if not smiles or smiles == "C":  # Avoid trivial molecules
            return 0.1
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
                
            # Compute Lipinski properties
            lipinski = self.compute_lipinski_properties(mol)
            admet = self.compute_admet_properties(mol)
            
            # Lipinski Rule of 5 compliance (each rule = 0.25 points)
            lipinski_score = 0.0
            lipinski_score += 0.25 if lipinski['mw'] <= 500 else 0.0
            lipinski_score += 0.25 if lipinski['logp'] <= 5 else 0.0  
            lipinski_score += 0.25 if lipinski['hbd'] <= 5 else 0.0
            lipinski_score += 0.25 if lipinski['hba'] <= 10 else 0.0
            
            # Additional ADMET factors
            tpsa_score = 1.0 if admet['tpsa'] <= 140 else max(0.0, 1.0 - (admet['tpsa'] - 140) / 140)
            rotbond_score = 1.0 if admet['rotatable_bonds'] <= 10 else max(0.0, 1.0 - (admet['rotatable_bonds'] - 10) / 10)
            
            # QED and SA scores (already normalized)
            qed_score = admet['qed']
            sa_score = admet['sa_score']
            
            # PAINS penalty (invert so lower alerts = higher score)
            pains_score = 1.0 - admet['pains_alerts']
            
            # Weighted combination of all factors
            final_score = (
                lipinski_score * 0.3 +        # 30% - Drug-likeness fundamentals
                tpsa_score * 0.15 +           # 15% - Membrane permeability  
                rotbond_score * 0.1 +         # 10% - Flexibility
                qed_score * 0.25 +            # 25% - Overall drug-likeness
                sa_score * 0.1 +              # 10% - Synthetic accessibility
                pains_score * 0.1             # 10% - Toxicity avoidance
            )
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"Error scoring molecule {smiles}: {e}")
            return 0.0


class RewardFunction:
    """
    Combined reward function for molecular generation RL fine-tuning.
    Combines DTI binding affinity and ADMET properties.
    """
    
    def __init__(
        self,
        dti_model: nn.Module,
        drug_encoder: nn.Module,
        beta_dti: float = 1.0,
        beta_admet: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize reward function.
        
        Args:
            dti_model: Trained DTI prediction model (expects drug_emb, protein_emb)
            drug_encoder: GNN encoder to convert molecular graphs to drug embeddings
            beta_dti: Weight for DTI component
            beta_admet: Weight for ADMET component  
            device: Device to run computations on
        """
        self.dti_model = dti_model
        self.drug_encoder = drug_encoder
        self.beta_dti = beta_dti
        self.beta_admet = beta_admet
        self.device = device or torch.device('cpu')
        
        # Put models on device and set to eval mode
        self.dti_model = self.dti_model.to(self.device).eval()
        self.drug_encoder = self.drug_encoder.to(self.device).eval()
        
        # Initialize ADMET scorer
        self.admet_scorer = ADMETScorer() if RDKIT_AVAILABLE else None
        
        # Statistics for normalization (will be updated during training)
        self.dti_stats = {'mean': 0.5, 'std': 0.25}
        self.admet_stats = {'mean': 0.5, 'std': 0.25}
        
        # Moving averages for online normalization
        self.dti_ema = 0.5
        self.admet_ema = 0.5
        self.ema_decay = 0.99
    
    def compute_dti_scores(
        self, 
        molecule_graphs: List[Data], 
        protein_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DTI binding affinity scores.
        
        Args:
            molecule_graphs: List of molecular graphs
            protein_embeddings: Protein embeddings [batch_size, protein_dim]
            
        Returns:
            DTI scores [batch_size] in range [0,1]
        """
        if len(molecule_graphs) == 0:
            return torch.zeros(0, device=self.device)
            
        try:
            # Convert molecular graphs to drug embeddings using GNN encoder
            mol_batch = Batch.from_data_list(molecule_graphs).to(self.device)
            
            self.drug_encoder.eval()
            with torch.no_grad():
                # Get 256D drug embeddings
                drug_embeddings = self.drug_encoder(
                    mol_batch.x, 
                    mol_batch.edge_index, 
                    mol_batch.edge_attr, 
                    mol_batch.batch
                )
            
            # Ensure protein embeddings are on correct device and have right shape
            protein_embeddings = protein_embeddings.to(self.device)
            if protein_embeddings.size(0) != drug_embeddings.size(0):
                # If we have multiple molecules per protein, expand protein embeddings
                molecules_per_protein = drug_embeddings.size(0) // protein_embeddings.size(0)
                protein_embeddings = protein_embeddings.repeat_interleave(molecules_per_protein, dim=0)
            
            # Get DTI predictions using your actual DTI model interface
            self.dti_model.eval()
            with torch.no_grad():
                # Your DTI model expects (drug_emb, protein_emb) and outputs logits
                dti_logits = self.dti_model(drug_embeddings, protein_embeddings)
                
                # Convert logits to probabilities [0,1]
                dti_scores = torch.sigmoid(dti_logits)
                
            return dti_scores
            
        except Exception as e:
            print(f"Error computing DTI scores: {e}")
            # Return neutral scores as fallback
            return torch.full((len(molecule_graphs),), 0.5, device=self.device)
    
    def compute_admet_scores(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Compute ADMET scores for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            ADMET scores [batch_size] in range [0,1]
        """
        if self.admet_scorer is None:
            print("Warning: ADMET scorer not available, returning neutral scores")
            return torch.full((len(smiles_list),), 0.5, device=self.device)
            
        admet_scores = []
        for smiles in smiles_list:
            score = self.admet_scorer.score_molecule(smiles) if smiles else 0.0
            admet_scores.append(score)
            
        return torch.tensor(admet_scores, dtype=torch.float, device=self.device)
    
    def update_normalization_stats(
        self, 
        dti_scores: torch.Tensor, 
        admet_scores: torch.Tensor
    ):
        """Update normalization statistics using exponential moving average."""
        if len(dti_scores) > 0:
            dti_mean = dti_scores.mean().item()
            self.dti_ema = self.ema_decay * self.dti_ema + (1 - self.ema_decay) * dti_mean
            
        if len(admet_scores) > 0:
            admet_mean = admet_scores.mean().item()
            self.admet_ema = self.ema_decay * self.admet_ema + (1 - self.ema_decay) * admet_mean
    
    def compute_reward(
        self,
        molecule_graphs: List[Data],
        protein_embeddings: torch.Tensor, 
        smiles_list: List[str],
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined reward for generated molecules.
        
        Args:
            molecule_graphs: List of molecular graphs
            protein_embeddings: Protein embeddings [batch_size, protein_dim]
            smiles_list: List of SMILES strings
            normalize: Whether to apply normalization
            
        Returns:
            Dictionary with 'total', 'dti', and 'admet' reward tensors
        """
        # Ensure consistent batch size
        batch_size = len(molecule_graphs)
        if protein_embeddings.size(0) != batch_size:
            # Expand protein embeddings if needed (multiple molecules per protein)
            protein_embeddings = protein_embeddings.repeat_interleave(
                batch_size // protein_embeddings.size(0), dim=0
            )
        
        # Compute individual scores
        dti_scores = self.compute_dti_scores(molecule_graphs, protein_embeddings)
        admet_scores = self.compute_admet_scores(smiles_list)
        
        # Update running statistics
        self.update_normalization_stats(dti_scores, admet_scores)
        
        # Optional normalization (centered around 0)
        if normalize:
            dti_normalized = dti_scores - self.dti_ema
            admet_normalized = admet_scores - self.admet_ema
        else:
            dti_normalized = dti_scores
            admet_normalized = admet_scores
        
        # Compute weighted combination
        total_reward = (
            self.beta_dti * dti_normalized + 
            self.beta_admet * admet_normalized
        )
        
        return {
            'total': total_reward,
            'dti': dti_scores,
            'admet': admet_scores,
            'dti_normalized': dti_normalized,
            'admet_normalized': admet_normalized
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get current normalization statistics."""
        return {
            'dti_ema': self.dti_ema,
            'admet_ema': self.admet_ema,
            'beta_dti': self.beta_dti,
            'beta_admet': self.beta_admet
        }
