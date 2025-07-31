"""
Fixed sampling method for ConditionalGraphDiffusion
This addresses the key issues: batch handling, edge sparsification, and realistic molecular structures
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, List
import math

def sample_fixed(
    self,
    protein_embeddings: torch.Tensor,
    num_atoms: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    edge_threshold: float = 0.3,  # NEW: Threshold for edge pruning
    max_degree: int = 4,  # NEW: Maximum degree per atom (realistic for molecules)
    return_intermediates: bool = False
) -> Batch:
    """
    FIXED sample method that generates realistic molecular graphs.
    
    Args:
        protein_embeddings: Protein conditioning [batch_size, protein_dim]
        num_atoms: Number of atoms per molecule [batch_size] 
        device: Device to run on
        edge_threshold: Probability threshold for keeping edges (0.0-1.0)
        max_degree: Maximum degree per atom for chemical realism
        return_intermediates: Whether to return generation intermediates
        
    Returns:
        Generated molecular graphs with realistic connectivity
    """
    if device is None:
        device = protein_embeddings.device
        
    batch_size = protein_embeddings.size(0)
    
    if num_atoms is None:
        # Sample number of atoms 
        num_atoms = torch.randint(5, self.max_atoms, (batch_size,), device=device)
    
    # FIX 1: Create separate molecules instead of single merged graph
    molecules = []
    
    for mol_idx in range(batch_size):
        n_atoms = num_atoms[mol_idx].item()
        protein_emb = protein_embeddings[mol_idx:mol_idx+1]  # Keep as [1, protein_dim]
        
        # Initialize molecule with random features
        atom_features = torch.randn(n_atoms, self.atom_feature_dim, device=device)
        coordinates = torch.randn(n_atoms, 3, device=device)
        
        # Create initial fully connected graph for this molecule
        source = torch.arange(n_atoms, device=device)
        target = torch.arange(n_atoms, device=device)
        edge_index = torch.cartesian_prod(source, target).t()
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Initialize edge attributes
        num_edges = edge_index.size(1)
        edge_attr = torch.randn(num_edges, self.edge_feature_dim, device=device)
        
        # Create batch assignment (all atoms belong to molecule 0 since we process one at a time)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
        
        # Create graph for this molecule
        mol_graph = Data(
            x=atom_features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )
        
        # Convert to batch for processing
        noisy_graph = Batch.from_data_list([mol_graph])
        
        # Reverse diffusion process for this molecule
        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                timestep = torch.tensor([t], device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.forward(noisy_graph, timestep, protein_emb)
                
                # DDPM denoising step (same as before but with tensor fixes)
                if t > 0:
                    alpha_t = 1.0 - (0.0001 + (0.02 - 0.0001) * t / self.timesteps)
                    alpha_t = 1.0 - alpha_t
                    alpha_bar_t = alpha_t
                    beta_t = 1.0 - alpha_t
                    
                    # Convert to tensors
                    alpha_t = torch.tensor(alpha_t, device=device)
                    alpha_bar_t = torch.tensor(alpha_bar_t, device=device)
                    beta_t = torch.tensor(beta_t, device=device)
                    
                    # Denoise
                    denoised_atoms = (noisy_graph.x - 
                                    (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['atom_pred']
                                    ) / torch.sqrt(alpha_t)
                    denoised_coords = (noisy_graph.pos - 
                                     (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['coord_pred']
                                     ) / torch.sqrt(alpha_t)
                    denoised_edges = (noisy_graph.edge_attr - 
                                    (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['edge_pred']
                                    ) / torch.sqrt(alpha_t)
                    
                    # Add noise for stochastic sampling
                    sigma_t = torch.sqrt(beta_t)
                    denoised_atoms += sigma_t * torch.randn_like(denoised_atoms)
                    denoised_coords += sigma_t * torch.randn_like(denoised_coords)
                    denoised_edges += sigma_t * torch.randn_like(denoised_edges)
                    
                    # FIX 2: Add feature clipping to prevent NaN
                    denoised_atoms = torch.clamp(denoised_atoms, min=-10, max=10)
                    denoised_coords = torch.clamp(denoised_coords, min=-50, max=50)
                    denoised_edges = torch.clamp(denoised_edges, min=-10, max=10)
                    
                    noisy_graph.x = denoised_atoms
                    noisy_graph.pos = denoised_coords
                    noisy_graph.edge_attr = denoised_edges
                    
                else:
                    # Final step
                    alpha_t = torch.tensor(1.0 - 0.0001, device=device)
                    alpha_bar_t = alpha_t
                    beta_t = torch.tensor(0.0001, device=device)
                    
                    final_atoms = (noisy_graph.x - 
                                 (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['atom_pred']
                                 ) / torch.sqrt(alpha_t)
                    final_coords = (noisy_graph.pos - 
                                  (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['coord_pred']
                                  ) / torch.sqrt(alpha_t)
                    final_edges = (noisy_graph.edge_attr - 
                                 (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise['edge_pred']
                                 ) / torch.sqrt(alpha_t)
                    
                    # Final clipping
                    noisy_graph.x = torch.clamp(final_atoms, min=-10, max=10)
                    noisy_graph.pos = torch.clamp(final_coords, min=-50, max=50)
                    noisy_graph.edge_attr = torch.clamp(final_edges, min=-10, max=10)
        
        # FIX 3: Add edge sparsification based on edge features
        edge_scores = noisy_graph.edge_attr.norm(dim=1)  # Use L2 norm of edge features as strength
        edge_probs = torch.sigmoid(edge_scores - edge_scores.mean())  # Center around mean
        
        # Keep edges above threshold
        keep_edges = edge_probs > edge_threshold
        
        # FIX 4: Enforce maximum degree constraint
        if max_degree > 0:
            degrees = torch.zeros(n_atoms, device=device)
            final_edge_mask = torch.zeros_like(keep_edges, dtype=torch.bool)
            
            # Sort edges by probability (keep strongest edges)
            edge_priorities = edge_probs[keep_edges]
            kept_edge_indices = torch.where(keep_edges)[0]
            
            if len(kept_edge_indices) > 0:
                sorted_indices = torch.argsort(edge_priorities, descending=True)
                
                for idx in sorted_indices:
                    edge_idx = kept_edge_indices[idx]
                    src, tgt = noisy_graph.edge_index[:, edge_idx]
                    
                    if degrees[src] < max_degree and degrees[tgt] < max_degree:
                        final_edge_mask[edge_idx] = True
                        degrees[src] += 1
                        degrees[tgt] += 1
        else:
            final_edge_mask = keep_edges
        
        # Apply edge mask
        if final_edge_mask.sum() > 0:
            noisy_graph.edge_index = noisy_graph.edge_index[:, final_edge_mask]
            noisy_graph.edge_attr = noisy_graph.edge_attr[final_edge_mask]
        else:
            # Ensure at least some connectivity - keep top 10% of edges
            n_keep = max(1, int(0.1 * len(edge_probs)))
            top_edges = torch.topk(edge_probs, n_keep).indices
            noisy_graph.edge_index = noisy_graph.edge_index[:, top_edges]
            noisy_graph.edge_attr = noisy_graph.edge_attr[top_edges]
        
        # Convert back to single molecule and add to list
        molecule = noisy_graph.to_data_list()[0]
        molecules.append(molecule)
    
    # FIX 1 (continued): Create proper batch from separate molecules
    return Batch.from_data_list(molecules)

