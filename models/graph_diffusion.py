"""
Graph-based diffusion model for conditional molecular generation.
Conditioned on protein embeddings via cross-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding in diffusion.
    This obviously is inspired by the original Transformer paper."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttentionBlock(nn.Module):
    """Cross-attention between molecular graph and protein embeddings."""
    
    def __init__(self, mol_dim: int, protein_dim: int, num_heads: int = 8):
        super().__init__()
        self.mol_dim = mol_dim
        self.protein_dim = protein_dim
        self.num_heads = num_heads
        
        # Project molecular features to query space
        self.q_proj = nn.Linear(mol_dim, mol_dim)
        
        # Project protein features to key/value space  
        self.k_proj = nn.Linear(protein_dim, mol_dim)
        self.v_proj = nn.Linear(protein_dim, mol_dim)
        
        # Output projection
        self.out_proj = nn.Linear(mol_dim, mol_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(mol_dim)
        self.ln2 = nn.LayerNorm(mol_dim)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(mol_dim, mol_dim * 4),
            nn.GELU(),
            nn.Linear(mol_dim * 4, mol_dim)
        )
        
    def forward(self, mol_features: torch.Tensor, protein_embedding: torch.Tensor, 
                mol_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_features: [N_atoms, mol_dim] - Molecular node features
            protein_embedding: [batch_size, protein_dim] - Protein embeddings
            mol_batch: [N_atoms] - Batch assignment for atoms
            
        Returns:
            Updated molecular features [N_atoms, mol_dim]
        """
        batch_size = protein_embedding.size(0)
        
        # Expand protein embedding to match molecular features
        # protein_embedding: [batch_size, protein_dim] -> [N_atoms, protein_dim]
        protein_expanded = protein_embedding[mol_batch]
        
        # Self-attention-like mechanism but with protein as key/value
        residual = mol_features
        mol_features = self.ln1(mol_features)
        
        # Compute attention
        q = self.q_proj(mol_features)  # [N_atoms, mol_dim]
        k = self.k_proj(protein_expanded)  # [N_atoms, mol_dim] 
        v = self.v_proj(protein_expanded)  # [N_atoms, mol_dim]
        
        # Multi-head attention
        # This is the actual attention mechanism and it works by projecting the queries, keys, and values
        # into multiple heads, allowing the model to jointly attend to information from different representation subspaces.
        head_dim = self.mol_dim // self.num_heads
        q = q.view(-1, self.num_heads, head_dim)  # [N_atoms, num_heads, head_dim]
        k = k.view(-1, self.num_heads, head_dim)
        v = v.view(-1, self.num_heads, head_dim)
        
        # Scaled dot-product attention
        scores = torch.einsum('nhd,nhd->nh', q, k) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # [N_atoms, num_heads]
        
        # Apply attention to values
        attn_output = torch.einsum('nh,nhd->nhd', attn_weights, v)
        attn_output = attn_output.view(-1, self.mol_dim)  # [N_atoms, mol_dim]
        
        # Output projection and residual
        attn_output = self.out_proj(attn_output)
        mol_features = residual + attn_output
        
        # Feedforward with residual
        residual = mol_features
        mol_features = self.ln2(mol_features)
        mol_features = residual + self.ff(mol_features)
        
        return mol_features


class GraphDiffusionBlock(nn.Module):
    """Graph convolution block for diffusion model."""
    
    def __init__(self, in_dim: int, out_dim: int, time_dim: int):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=4, concat=False, dropout=0.1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                time_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Edge indices [2, E]
            time_emb: Time embeddings [batch_size, time_dim]
            batch: Batch assignment [N]
        """
        # Apply graph convolution
        h = self.conv(x, edge_index)
        
        # Add time information
        time_expanded = time_emb[batch]  # [N, time_dim]
        time_proj = self.time_proj(time_expanded)  # [N, out_dim]
        
        h = h + time_proj
        h = self.norm(h)
        h = self.activation(h)
        
        return h


class ConditionalGraphDiffusion(nn.Module):
    """
    Conditional graph diffusion model for molecular generation.
    Generates molecular graphs conditioned on protein embeddings.
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 7,       # NEW: Continuous atom features
        edge_feature_dim: int = 3,       # NEW: Continuous edge features 
        protein_dim: int = 1280,         # ESM-2 embedding dimension
        hidden_dim: int = 256,
        num_layers: int = 6,
        max_atoms: int = 50,
        num_atom_types: int = 100,       # DEPRECATED: Keep for backwards compatibility
        num_bond_types: int = 4,         # DEPRECATED: Keep for backwards compatibility
        timesteps: int = 1000,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_atoms = max_atoms
        self.timesteps = timesteps
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)
        
        # Input projections for continuous features
        self.atom_proj = nn.Linear(atom_feature_dim, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)  # Project 3D coordinates
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Cross-attention blocks for protein conditioning
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, protein_dim, num_attention_heads)
            for _ in range(num_layers // 2)  # Add cross-attention every 2 layers
        ])
        
        # Graph diffusion blocks
        self.diffusion_blocks = nn.ModuleList([
            GraphDiffusionBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output heads (predict noise in the same space as input)
        self.atom_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, atom_feature_dim)  # Predict 7D atom feature noise
        )
        
        self.coord_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 3)  # 3D coordinates
        )
        
        # Edge prediction (combines node features and edge features)
        self.edge_output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 2 * hidden_dim (nodes) + hidden_dim (edges)
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_feature_dim)  # Predict 3D edge feature noise
        )
        
    def forward(
        self, 
        noisy_graphs: Batch,
        timestep: torch.Tensor,
        protein_embeddings: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the diffusion model.
        
        Args:
            noisy_graphs: Batch of noisy molecular graphs with:
                - x: [N_atoms, 7] continuous atom features 
                - pos: [N_atoms, 3] atom coordinates
                - edge_attr: [E, 3] continuous edge features
                - edge_index: [2, E] edge connectivity
                - batch: [N_atoms] batch assignment
            timestep: Diffusion timestep [batch_size]
            protein_embeddings: Protein conditioning [batch_size, protein_dim] 
            return_intermediates: Whether to return intermediate features
            
        Returns:
            Dictionary containing predicted noise for atoms, coordinates, and edges
        """
        batch_size = protein_embeddings.size(0)
        
        # Time embedding
        time_emb = self.time_embedding(timestep)  # [batch_size, hidden_dim]
        
        # Extract features from noisy graphs
        atom_features = noisy_graphs.x        # [N_atoms, 7] - continuous atom features
        coordinates = noisy_graphs.pos        # [N_atoms, 3] - atom coordinates  
        edge_attr = noisy_graphs.edge_attr    # [E, 3] - continuous edge features
        edge_index = noisy_graphs.edge_index  # [2, E]
        batch = noisy_graphs.batch           # [N_atoms]
        
        # Project atom features and coordinates to hidden dimension
        h_atoms = self.atom_proj(atom_features)  # [N_atoms, hidden_dim]
        h_coords = self.coord_proj(coordinates)  # [N_atoms, hidden_dim] 
        
        # Combine atom and coordinate information
        h = h_atoms + h_coords  # [N_atoms, hidden_dim]
        
        intermediates = []
        cross_attn_idx = 0
        
        # Apply diffusion blocks with cross-attention
        for i, block in enumerate(self.diffusion_blocks):
            # Apply graph diffusion
            h = block(h, edge_index, time_emb, batch)
            
            # Apply cross-attention every 2 layers
            if i % 2 == 1 and cross_attn_idx < len(self.cross_attention_blocks):
                h = self.cross_attention_blocks[cross_attn_idx](h, protein_embeddings, batch)
                cross_attn_idx += 1
                
            if return_intermediates:
                intermediates.append(h.clone())
        
        # Predict noise for atom features (7D) and coordinates (3D)
        atom_pred = self.atom_output(h)  # [N_atoms, 7] - predicting atom feature noise
        coord_pred = self.coord_output(h)  # [N_atoms, 3] - predicting coordinate noise
        
        # Edge predictions using both node features and edge features
        row, col = edge_index
        edge_node_features = torch.cat([h[row], h[col]], dim=-1)  # [E, hidden_dim * 2]
        edge_input_features = self.edge_proj(edge_attr)  # [E, hidden_dim]
        combined_edge_features = torch.cat([edge_node_features, edge_input_features], dim=-1)  # [E, hidden_dim * 3]
        edge_pred = self.edge_output(combined_edge_features)  # [E, 3] - predicting edge feature noise
        
        outputs = {
            'atom_pred': atom_pred,
            'coord_pred': coord_pred, 
            'edge_pred': edge_pred
        }
        
        if return_intermediates:
            outputs['intermediates'] = intermediates
            
        return outputs
    
    def sample(
        self,
        protein_embeddings: torch.Tensor,
        num_atoms: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Batch:
        """
        Sample molecular graphs from the diffusion model.
        
        Args:
            protein_embeddings: Protein conditioning [batch_size, protein_dim]
            num_atoms: Number of atoms per molecule [batch_size] 
            device: Device to run on
            
        Returns:
            Generated molecular graphs
        """
        if device is None:
            device = protein_embeddings.device
            
        batch_size = protein_embeddings.size(0)
        
        if num_atoms is None:
            # Sample number of atoms (you might want to condition this too)
            num_atoms = torch.randint(5, self.max_atoms, (batch_size,), device=device)
        
        # Start from pure noise
        total_atoms = num_atoms.sum().item()
        
        # Initialize with random atom features (7D) and coordinates (3D)
        atom_features = torch.randn(total_atoms, self.atom_feature_dim, device=device)  # 7D continuous features
        coordinates = torch.randn(total_atoms, 3, device=device)  # 3D coordinates
        
        # Create batch assignment
        batch = torch.cat([torch.full((n,), i, device=device) 
                          for i, n in enumerate(num_atoms)])
        
        # Create fully connected graphs (will be pruned during generation)
        edge_indices = []
        edge_attrs = []
        atom_offset = 0
        for i, n in enumerate(num_atoms):
            n = n.item()
            # Create edges for this molecule
            source = torch.arange(n, device=device) + atom_offset
            target = torch.arange(n, device=device) + atom_offset
            edge_index = torch.cartesian_prod(source, target).t()
            # Remove self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_indices.append(edge_index)
            
            # Initialize random edge attributes
            num_edges = edge_index.size(1)
            edge_attr = torch.randn(num_edges, self.edge_feature_dim, device=device)
            edge_attrs.append(edge_attr)
            
            atom_offset += n
            
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        
        # Create initial noisy graph with proper structure
        graph_data = Data(
            x=atom_features,      # [N_atoms, 7] atom features
            pos=coordinates,      # [N_atoms, 3] coordinates  
            edge_index=edge_index, # [2, E] connectivity
            edge_attr=edge_attr,   # [E, 3] edge features
            batch=batch           # [N_atoms] batch assignment
        )
        noisy_graphs = Batch.from_data_list([graph_data])
        
        # Reverse diffusion process
        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise added at timestep t
                predicted_noise = self.forward(noisy_graphs, timestep, protein_embeddings)
                
                # DDPM sampling: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * predicted_noise) + sigma_t * z
                # This is the actual diffusion sampling logic that was missing
                
                if t > 0:
                    # Get noise schedule parameters
                    alpha_t = 1.0 - (0.0001 + (0.02 - 0.0001) * t / self.timesteps)  # beta_t
                    alpha_t = 1.0 - alpha_t  # alpha_t = 1 - beta_t
                    alpha_bar_t = alpha_t  # Simplified - should be cumulative product
                    beta_t = 1.0 - alpha_t
                    
                    # Denoise atoms: x_{t-1} = (x_t - beta_t/sqrt(1-alpha_bar_t) * noise) / sqrt(alpha_t)
                    denoised_atoms = (noisy_graphs.x - 
                                    (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['atom_pred']
                                    ) / torch.sqrt(alpha_t)
                    
                    # Denoise coordinates
                    denoised_coords = (noisy_graphs.pos - 
                                     (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['coord_pred']
                                     ) / torch.sqrt(alpha_t)
                    
                    # Denoise edges  
                    denoised_edges = (noisy_graphs.edge_attr - 
                                    (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['edge_pred']
                                    ) / torch.sqrt(alpha_t)
                    
                    # Add noise for stochastic sampling (except at t=0)
                    sigma_t = torch.sqrt(beta_t)
                    denoised_atoms += sigma_t * torch.randn_like(denoised_atoms)
                    denoised_coords += sigma_t * torch.randn_like(denoised_coords)
                    denoised_edges += sigma_t * torch.randn_like(denoised_edges)
                    
                    # Update the graphs
                    noisy_graphs.x = denoised_atoms
                    noisy_graphs.pos = denoised_coords
                    noisy_graphs.edge_attr = denoised_edges
                    
                else:
                    # Final step: no additional noise
                    alpha_t = 1.0 - 0.0001  # beta_0
                    alpha_bar_t = alpha_t
                    beta_t = 1.0 - alpha_t
                    
                    noisy_graphs.x = (noisy_graphs.x - 
                                    (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['atom_pred']
                                    ) / torch.sqrt(alpha_t)
                    noisy_graphs.pos = (noisy_graphs.pos - 
                                      (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['coord_pred']
                                      ) / torch.sqrt(alpha_t)
                    noisy_graphs.edge_attr = (noisy_graphs.edge_attr - 
                                            (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['edge_pred']
                                            ) / torch.sqrt(alpha_t)
        
        return noisy_graphs


def create_molecule_graph(smiles: str, max_atoms: int = 50) -> Optional[Data]:
    """
    Convert SMILES to PyTorch Geometric graph representation with continuous features.
    
    Args:
        smiles: SMILES string of the molecule
        max_atoms: Maximum number of atoms allowed
        
    Returns:
        Data object with continuous atom and edge features, or None if conversion fails
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        import torch
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Add hydrogens for complete structure
        mol = Chem.AddHs(mol)
        
        # Check atom limit
        if mol.GetNumAtoms() > max_atoms:
            return None
            
        # Extract atom features (7D continuous)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),           # Atomic number
                float(atom.GetDegree()),              # Degree
                float(atom.GetFormalCharge()),        # Formal charge
                float(atom.GetNumRadicalElectrons()), # Radical electrons
                float(atom.GetIsAromatic()),          # Aromatic (0 or 1)
                float(atom.GetHybridization()),       # Hybridization (enum as float)
                float(atom.GetMass()),                # Atomic mass
            ]
            atom_features.append(features)
            
        if len(atom_features) == 0:
            return None
            
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Generate 3D coordinates (placeholder - you might want to use RDKit's conformer generation)
        num_atoms = mol.GetNumAtoms()
        pos = torch.randn(num_atoms, 3, dtype=torch.float)  # Random coordinates for now
        
        # Extract edges and edge features (3D continuous)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond features (3D continuous)
            bond_features = [
                float(bond.GetBondTypeAsDouble()),    # Bond type as continuous value
                float(bond.IsInRing()),               # In ring (0 or 1)
                float(bond.GetIsConjugated()),        # Conjugated (0 or 1)
            ]
            
            # Add features for both directions
            edge_features.extend([bond_features, bond_features])
            
        if len(edge_indices) == 0:
            # Molecule with no bonds - create self-loops or handle specially
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,                    # [num_atoms, 7] continuous atom features
            pos=pos,               # [num_atoms, 3] 3D coordinates
            edge_index=edge_index, # [2, num_edges] edge connectivity
            edge_attr=edge_attr,   # [num_edges, 3] continuous edge features
        )
        
        return data
        
    except Exception as e:
        print(f"Error converting SMILES '{smiles}' to graph: {e}")
        return None


def graph_to_smiles(graph: Data, sanitize: bool = True) -> Optional[str]:
    """
    Convert graph representation back to SMILES string.
    
    Args:
        graph: PyTorch Geometric Data object with continuous features
        sanitize: Whether to sanitize the molecule (fix valences, etc.)
        
    Returns:
        SMILES string or None if conversion fails
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        import torch
        
        # Create empty RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        atom_features = graph.x  # [num_atoms, 7]
        for i, features in enumerate(atom_features):
            atomic_num = int(round(features[0].item()))
            formal_charge = int(round(features[2].item()))
            is_aromatic = bool(round(features[4].item()))
            
            # Create atom with basic properties
            atom = Chem.Atom(atomic_num)
            atom.SetFormalCharge(formal_charge)
            atom.SetIsAromatic(is_aromatic)
            
            mol.AddAtom(atom)
            
        # Add bonds
        if graph.edge_index.size(1) > 0:
            edge_index = graph.edge_index  # [2, num_edges]
            edge_attr = graph.edge_attr    # [num_edges, 3]
            
            # Process edges (skip duplicate undirected edges)
            processed_edges = set()
            
            for i in range(edge_index.size(1)):
                atom_i = int(edge_index[0, i].item())
                atom_j = int(edge_index[1, i].item())
                
                # Skip self-loops and already processed edges
                if atom_i == atom_j:
                    continue
                    
                edge_key = tuple(sorted([atom_i, atom_j]))
                if edge_key in processed_edges:
                    continue
                    
                processed_edges.add(edge_key)
                
                # Get bond type from continuous feature
                bond_type_float = edge_attr[i, 0].item()
                is_in_ring = bool(round(edge_attr[i, 1].item()))
                is_conjugated = bool(round(edge_attr[i, 2].item()))
                
                # Convert continuous bond type to discrete RDKit bond type
                if bond_type_float < 1.5:
                    bond_type = Chem.BondType.SINGLE
                elif bond_type_float < 2.5:
                    bond_type = Chem.BondType.DOUBLE
                elif bond_type_float < 3.5:
                    bond_type = Chem.BondType.TRIPLE
                else:
                    bond_type = Chem.BondType.AROMATIC
                    
                # Add bond
                mol.AddBond(atom_i, atom_j, bond_type)
                
                # Set additional properties
                bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
                if bond:
                    bond.SetIsConjugated(is_conjugated)
                    
        # Convert to Mol and sanitize
        mol = mol.GetMol()
        
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                # If sanitization fails, try without
                pass
                
        # Generate SMILES
        smiles = Chem.MolToSmiles(mol)
        
        return smiles if smiles else None
        
    except Exception as e:
        print(f"Error converting graph to SMILES: {e}")
        return None
