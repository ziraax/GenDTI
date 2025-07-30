"""
Training script for conditional molecular diffusion model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
import yaml
import os
import sys
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.graph_diffusion import ConditionalGraphDiffusion
from models.diffusion_utils import DiffusionSchedule, GraphDiffusionProcess
from data.molecular_dataset import (
    load_chembl_data, 
    create_molecular_dataloader,
    create_conditional_dataloader
)
from utils.save_utils import SaveManager

# Setup logging
# TODO: Not properly working right now
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """Trainer for conditional molecular diffusion model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'molecular-diffusion'),
                name=config.get('experiment_name'),
                config=config
            )
        
        # Setup save manager
        self.save_manager = SaveManager(
            save_dir=config['save_dir'],
            experiment_name=config.get('experiment_name', 'diffusion_experiment')
        )
        
        # Initialize model components
        self.setup_model()
        self.setup_diffusion()
        self.setup_optimizer()
        self.setup_data()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
    def setup_model(self):
        """Initialize the diffusion model."""
        model_config = self.config['model']
        
        self.model = ConditionalGraphDiffusion(
            atom_feature_dim=model_config.get('atom_feature_dim', 7),
            edge_feature_dim=model_config.get('edge_feature_dim', 3),
            protein_dim=model_config.get('protein_dim', 1280),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 6),
            max_atoms=model_config.get('max_atoms', 50),
            num_atom_types=model_config.get('num_atom_types', 100),  # Backwards compatibility
            num_bond_types=model_config.get('num_bond_types', 4),   # Backwards compatibility
            timesteps=model_config.get('timesteps', 1000),
            num_attention_heads=model_config.get('num_attention_heads', 8)
        ).to(self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def setup_diffusion(self):
        """Initialize diffusion process."""
        diffusion_config = self.config.get('diffusion', {})
        
        schedule = DiffusionSchedule(
            timesteps=diffusion_config.get('timesteps', 1000),
            beta_start=diffusion_config.get('beta_start', 0.0001),
            beta_end=diffusion_config.get('beta_end', 0.02),
            schedule_type=diffusion_config.get('schedule_type', 'cosine')
        )
        
        self.diffusion = GraphDiffusionProcess(
            schedule, 
            device=str(self.device),
            num_atom_types=self.config['model'].get('num_atom_types', 44),
            num_bond_types=self.config['model'].get('num_bond_types', 4)
        )
        
    def setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        optimizer_config = self.config['optimizer']
        
        if optimizer_config['name'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['name'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
        
        # Setup scheduler if specified
        scheduler_config = self.config.get('scheduler')
        if scheduler_config:
            if scheduler_config['name'].lower() == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config['epochs']
                )
            elif scheduler_config['name'].lower() == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 30),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
            
    def setup_data(self):
        """Setup data loaders."""
        data_config = self.config['data']
        
        if self.config['training_stage'] == 'pretraining':
            # Stage 1: Pretraining on ChEMBL
            logger.info("Setting up pretraining data (ChEMBL)")
            
            smiles_list = load_chembl_data(
                data_config['chembl_file'],
                sample_size=data_config.get('sample_size'),
                max_atoms=self.config['model'].get('max_atoms', 50)
            )
            
            self.train_loader = create_molecular_dataloader(
                smiles_list,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=data_config.get('num_workers', 4),
                max_atoms=self.config['model'].get('max_atoms', 50),
                cache_dir=data_config.get('cache_dir')
            )
            
            self.val_loader = None  # No validation for pretraining
            
        elif self.config['training_stage'] == 'conditional':
            # Stage 2: Conditional training on DTI data
            logger.info("Setting up conditional training data")
            
            self.train_loader = create_conditional_dataloader(
                data_config['train_triplets'],
                data_config['protein_embeddings'],
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=data_config.get('num_workers', 4),
                max_atoms=self.config['model'].get('max_atoms', 50),
                molecular_cache_dir=data_config.get('cache_dir')
            )
            
            self.val_loader = create_conditional_dataloader(
                data_config['val_triplets'],
                data_config['protein_embeddings'],
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=data_config.get('num_workers', 4),
                max_atoms=self.config['model'].get('max_atoms', 50),
                molecular_cache_dir=data_config.get('cache_dir')
            )
        else:
            raise ValueError(f"Unknown training stage: {self.config['training_stage']}")
        
        logger.info(f"Train loader: {len(self.train_loader)} batches")
        if self.val_loader:
            logger.info(f"Val loader: {len(self.val_loader)} batches")
    
    def train_step_pretraining(self, batch) -> Dict[str, float]:
        """Training step for pretraining stage (unconditional generation)."""
        graphs = batch.to(self.device)
        batch_size = graphs.batch.max().item() + 1
        
        # Sample timesteps
        t = torch.randint(0, self.diffusion.schedule.timesteps, (batch_size,), device=self.device)
        
        # Use dummy protein embeddings for pretraining
        protein_embeddings = torch.zeros(batch_size, self.config['model']['protein_dim'], device=self.device)
        
        # Compute loss
        losses = self.diffusion.p_losses(self.model, graphs, protein_embeddings, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        if self.config.get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train_step_conditional(self, batch) -> Dict[str, float]:
        """Training step for conditional generation."""
        graphs = batch['molecular_graphs'].to(self.device)
        protein_embeddings = batch['protein_embeddings'].to(self.device)
        batch_size = protein_embeddings.size(0)
        
        # Sample timesteps
        t = torch.randint(0, self.diffusion.schedule.timesteps, (batch_size,), device=self.device)
        
        # Compute loss
        losses = self.diffusion.p_losses(self.model, graphs, protein_embeddings, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        if self.config.get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def val_step_pretraining(self, batch) -> Dict[str, float]:
        """Validation step for pretraining stage (no backward pass)."""
        graphs = batch.to(self.device)
        batch_size = graphs.batch.max().item() + 1
        
        # Sample timesteps
        t = torch.randint(0, self.diffusion.schedule.timesteps, (batch_size,), device=self.device)
        
        # Use dummy protein embeddings for pretraining
        protein_embeddings = torch.zeros(batch_size, self.config['model']['protein_dim'], device=self.device)
        
        # Compute loss (no backward pass)
        losses = self.diffusion.p_losses(self.model, graphs, protein_embeddings, t)
        
        return {k: v.item() for k, v in losses.items()}
    
    def val_step_conditional(self, batch) -> Dict[str, float]:
        """Validation step for conditional generation (no backward pass)."""
        graphs = batch['molecular_graphs'].to(self.device)
        protein_embeddings = batch['protein_embeddings'].to(self.device)
        batch_size = protein_embeddings.size(0)
        
        # Sample timesteps
        t = torch.randint(0, self.diffusion.schedule.timesteps, (batch_size,), device=self.device)
        
        # Compute loss (no backward pass)
        losses = self.diffusion.p_losses(self.model, graphs, protein_embeddings, t)
        
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if self.config['training_stage'] == 'conditional':
                    losses = self.val_step_conditional(batch)
                else:
                    losses = self.val_step_pretraining(batch)
                
                # Accumulate losses
                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v
                num_batches += 1
        
        # Average losses
        avg_losses = {f"val_{k}": v / num_batches for k, v in total_losses.items()}
        
        self.model.train()
        return avg_losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Training step
            if self.config['training_stage'] == 'pretraining':
                losses = self.train_step_pretraining(batch)
            else:
                losses = self.train_step_conditional(batch)
            
            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            
            # Log to wandb
            if self.config.get('use_wandb') and self.global_step % self.config.get('log_every', 100) == 0:
                wandb.log({f"train_{k}": v for k, v in losses.items()}, step=self.global_step)
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training stage: {self.config['training_stage']}")
        logger.info(f"Epochs: {self.config['epochs']}")
        logger.info(f"Batch size: {self.config['batch_size']}")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Store current loss for checkpointing
            self.current_loss = train_losses.get('total_loss', 0.0)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch results
            epoch_metrics = {**train_losses, **val_losses}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in epoch_metrics.items()]))
            
            if self.config.get('use_wandb'):
                wandb.log(epoch_metrics, step=self.global_step)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch)
        
        # Save final model
        self.save_checkpoint(self.config['epochs'] - 1, is_final=True)
        logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint."""
        filename = f"diffusion_checkpoint_epoch_{epoch}.pt"
        if is_final:
            filename = "diffusion_model_final.pt"
        
        # Get current loss from last batch
        current_loss = getattr(self, 'current_loss', 0.0)
        
        self.save_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            loss=current_loss,
            metrics={'global_step': self.global_step},
            is_best=is_final
        )
        logger.info(f"Saved checkpoint: {filename}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train conditional molecular diffusion model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.name:
        config['experiment_name'] = args.name
    
    # Create trainer and start training
    trainer = DiffusionTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
