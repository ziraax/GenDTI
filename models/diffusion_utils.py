"""
Diffusion process utilities for molecular graph generation.
Implements forward and reverse diffusion processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Optional
import math


class DiffusionSchedule:
    """
    Diffusion noise schedule for graph generation.
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine"
    ):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Create noise schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


class GraphDiffusionProcess:
    """
    Graph diffusion process for molecular generation.
    Handles forward (noise addition) and reverse (denoising) processes.
    """
    
    def __init__(self, schedule: DiffusionSchedule, device: str = 'cpu', num_atom_types: int = 44, num_bond_types: int = 4):
        self.schedule = schedule
        self.device = device
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        
        # Move schedule tensors to device
        self.schedule.betas = self.schedule.betas.to(device)
        self.schedule.alphas = self.schedule.alphas.to(device)
        self.schedule.alphas_cumprod = self.schedule.alphas_cumprod.to(device)
        self.schedule.sqrt_alphas_cumprod = self.schedule.sqrt_alphas_cumprod.to(device)
        self.schedule.sqrt_one_minus_alphas_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod.to(device)
        self.schedule.posterior_variance = self.schedule.posterior_variance.to(device)
        
    def q_sample(
        self, 
        graphs: Batch, 
        t: torch.Tensor,
        noise: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Batch, Dict[str, torch.Tensor]]:
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to clean graphs according to timestep t.
        
        Args:
            graphs: Clean molecular graphs
            t: Timestep [batch_size]
            noise: Optional pre-computed noise
            
        Returns:
            Noisy graphs and the noise that was added
        """
        if noise is None:
            noise = self._sample_noise(graphs)
            
        # Get noise coefficients for this timestep
        sqrt_alpha_prod = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        
        # Expand coefficients to match graph dimensions
        batch_size = t.size(0)
        sqrt_alpha_expanded = sqrt_alpha_prod[graphs.batch]
        sqrt_one_minus_alpha_expanded = sqrt_one_minus_alpha_prod[graphs.batch]
        
        # Apply noise to node features (now continuous 7D atom features)
        noisy_x = (
            sqrt_alpha_expanded.unsqueeze(-1) * graphs.x + 
            sqrt_one_minus_alpha_expanded.unsqueeze(-1) * noise['atom_noise']
        )
        
        # Apply noise to coordinates
        noisy_pos = (
            sqrt_alpha_expanded.unsqueeze(-1) * graphs.pos + 
            sqrt_one_minus_alpha_expanded.unsqueeze(-1) * noise['coord_noise']
        )
        
        # Apply noise to edges (now continuous 3D edge features)
        if hasattr(graphs, 'edge_attr') and graphs.edge_attr is not None:
            # Expand coefficients for edges
            edge_batch = graphs.batch[graphs.edge_index[0]]  # Get batch assignment for each edge
            sqrt_alpha_edge = sqrt_alpha_prod[edge_batch]
            sqrt_one_minus_alpha_edge = sqrt_one_minus_alpha_prod[edge_batch]
            
            noisy_edge_attr = (
                sqrt_alpha_edge.unsqueeze(-1) * graphs.edge_attr + 
                sqrt_one_minus_alpha_edge.unsqueeze(-1) * noise['edge_noise']
            )
        else:
            noisy_edge_attr = None
        
        # Create noisy graph maintaining batch structure
        noisy_graphs = Batch(
            x=noisy_x,
            pos=noisy_pos,
            edge_index=graphs.edge_index,
            edge_attr=noisy_edge_attr,
            batch=graphs.batch
        )
        
        return noisy_graphs, noise
    
    def _sample_noise(self, graphs: Batch) -> Dict[str, torch.Tensor]:
        """Sample noise for atoms, coordinates, and edges."""
        device = graphs.x.device
        
        # Atom noise (continuous 7D features - use gaussian)
        atom_noise = torch.randn_like(graphs.x.float())
        
        # Coordinate noise (continuous - use gaussian)  
        coord_noise = torch.randn_like(graphs.pos)
        
        # Edge noise (continuous 3D features - use gaussian)
        if hasattr(graphs, 'edge_attr') and graphs.edge_attr is not None:
            edge_noise = torch.randn_like(graphs.edge_attr.float())
        else:
            # If no edge attributes, create dummy noise
            edge_noise = torch.randn(graphs.edge_index.size(1), 3, device=device)
            
        return {
            'atom_noise': atom_noise,
            'coord_noise': coord_noise, 
            'edge_noise': edge_noise
        }
    
    def _add_categorical_noise(
        self,
        categorical_data: torch.Tensor,
        sqrt_alpha: torch.Tensor,
        sqrt_one_minus_alpha: torch.Tensor,
        noise: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Add noise to categorical data (atom types, bond types).
        Uses categorical diffusion process.
        """
        if categorical_data is None:
            return None
            
        # Convert to one-hot if needed
        if categorical_data.dtype == torch.long:
            # Use provided num_classes or infer from data max
            if num_classes is None:
                num_classes = categorical_data.max().item() + 1
            one_hot = F.one_hot(categorical_data, num_classes).float()
        else:
            one_hot = categorical_data
            
        # Apply noise (simplified categorical diffusion)
        # sqrt_alpha and sqrt_one_minus_alpha are already expanded to match nodes
        # They have shape [num_nodes] and need to broadcast with [num_nodes, num_classes]
        noisy_probs = (
            sqrt_alpha.unsqueeze(-1) * one_hot + 
            sqrt_one_minus_alpha.unsqueeze(-1) * noise
        )
        
        # Normalize to valid probabilities
        noisy_probs = F.softmax(noisy_probs, dim=-1)
        
        return noisy_probs
    
    def p_losses(
        self,
        model: nn.Module,
        graphs: Batch,
        protein_embeddings: torch.Tensor,
        t: torch.Tensor,
        loss_type: str = "l2"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion losses.
        
        Args:
            model: Diffusion model
            graphs: Clean molecular graphs
            protein_embeddings: Protein conditioning
            t: Timestep
            loss_type: Type of loss ("l2", "l1", "huber")
            
        Returns:
            Dictionary of losses
        """
        # Sample noise and create noisy graphs
        noise = self._sample_noise(graphs)
        noisy_graphs, _ = self.q_sample(graphs, t, noise)
        
        # Predict noise
        predicted = model(noisy_graphs, t, protein_embeddings)
        
        # Compute losses
        losses = {}
        
        # Atom prediction loss
        if loss_type == "l2":
            atom_loss = F.mse_loss(predicted['atom_pred'], noise['atom_noise'])
        elif loss_type == "l1":
            atom_loss = F.l1_loss(predicted['atom_pred'], noise['atom_noise'])
        elif loss_type == "huber":
            atom_loss = F.smooth_l1_loss(predicted['atom_pred'], noise['atom_noise'])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        losses['atom_loss'] = atom_loss
        
        # Coordinate prediction loss
        coord_loss = F.mse_loss(predicted['coord_pred'], noise['coord_noise'])
        losses['coord_loss'] = coord_loss
        
        # Edge prediction loss
        edge_loss = F.mse_loss(predicted['edge_pred'], noise['edge_noise'])
        losses['edge_loss'] = edge_loss
        

        losses['total_loss'] = atom_loss + coord_loss + edge_loss
        
        return losses
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        graphs: Batch,
        protein_embeddings: torch.Tensor,
        t: torch.Tensor,
        t_index: int
    ) -> Batch:
        """
        Single step of reverse diffusion sampling.
        
        Args:
            model: Diffusion model
            graphs: Current noisy graphs
            protein_embeddings: Protein conditioning
            t: Current timestep
            t_index: Index of current timestep
            
        Returns:
            Slightly less noisy graphs
        """
        # Predict noise
        predicted = model(graphs, t, protein_embeddings)
        
        # Get diffusion parameters
        betas_t = self.schedule.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.schedule.alphas[t])
        
        # Expand to match graph dimensions
        sqrt_recip_alphas_expanded = sqrt_recip_alphas_t[graphs.batch]
        betas_expanded = betas_t[graphs.batch]
        sqrt_one_minus_alphas_cumprod_expanded = sqrt_one_minus_alphas_cumprod_t[graphs.batch]
        
        # Predict previous sample
        pred_original_sample_atoms = (
            sqrt_recip_alphas_expanded.unsqueeze(-1) * 
            (graphs.x - betas_expanded.unsqueeze(-1) * predicted['atom_pred'] / 
             sqrt_one_minus_alphas_cumprod_expanded.unsqueeze(-1))
        )
        
        pred_original_sample_coords = (
            sqrt_recip_alphas_expanded.unsqueeze(-1) * 
            (graphs.pos - betas_expanded.unsqueeze(-1) * predicted['coord_pred'] / 
             sqrt_one_minus_alphas_cumprod_expanded.unsqueeze(-1))
        )
        
        # Add noise if not the last step
        if t_index > 0:
            posterior_variance_t = self.schedule.posterior_variance[t]
            noise_atoms = torch.randn_like(pred_original_sample_atoms)
            noise_coords = torch.randn_like(pred_original_sample_coords)
            
            posterior_variance_expanded = posterior_variance_t[graphs.batch]
            pred_original_sample_atoms += torch.sqrt(posterior_variance_expanded).unsqueeze(-1) * noise_atoms
            pred_original_sample_coords += torch.sqrt(posterior_variance_expanded).unsqueeze(-1) * noise_coords
        
        # Create updated graph
        updated_graphs = Data(
            x=pred_original_sample_atoms,
            pos=pred_original_sample_coords,
            edge_index=graphs.edge_index,
            edge_attr=graphs.edge_attr,
            batch=graphs.batch
        )
        
        return Batch.from_data_list([updated_graphs])
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        protein_embeddings: torch.Tensor,
        num_atoms: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Batch:
        """
        Full reverse diffusion sampling loop.
        Generate molecular graphs from pure noise.
        
        Args:
            model: Diffusion model
            protein_embeddings: Protein conditioning
            num_atoms: Number of atoms per molecule
            device: Device to run on
            
        Returns:
            Generated molecular graphs
        """
        if device is None:
            device = protein_embeddings.device
            
        model.eval()
        
        # Start from pure noise
        graphs = model.sample(protein_embeddings, num_atoms, device)
        
        # Iteratively denoise
        for i in reversed(range(self.schedule.timesteps)):
            t = torch.full(
                (protein_embeddings.size(0),), 
                i, 
                device=device, 
                dtype=torch.long
            )
            
            graphs = self.p_sample(model, graphs, protein_embeddings, t, i)
            
        return graphs


# Utility functions for molecular graph processing
def preprocess_molecular_graphs(smiles_list: list) -> Batch:
    """
    Convert list of SMILES to batch of molecular graphs.
    This is a placeholder - you'll need to implement with RDKit.
    """
    # TODO: Implement SMILES to graph conversion
    pass


def postprocess_molecular_graphs(graphs: Batch) -> list:
    """
    Convert batch of molecular graphs back to SMILES.
    This is a placeholder - you'll need to implement with RDKit.
    """
    # TODO: Implement graph to SMILES conversion
    pass
