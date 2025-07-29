"""
DTI Model Trainer with advanced training features and clean separation of concerns.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, CosineAnnealingWarmRestarts
)
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional, Tuple
import logging

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dti_model import DTIModel
from data.utils.dti_dataset import DTIDataset


class DTITrainer:
    """Advanced trainer for DTI models with comprehensive training features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('inf') if config.get('monitor_mode', 'min') == 'min' else 0
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Setup experiment tracking
        if config.get('use_wandb', False):
            self.setup_wandb()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['log_dir'], 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb.init(
            project=self.config.get('wandb_project', 'dti-training'),
            config=self.config,
            name=self.config.get('experiment_name', None)
        )
    
    def build_model(self, drug_dim: int, protein_dim: int) -> DTIModel:
        """Build and return the DTI model."""
        model = DTIModel(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            fusion=self.config['model']['fusion'],
            proj_dim=self.config['model']['proj_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        self.logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build and return the optimizer."""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name'].lower()
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0),
                momentum=optimizer_config.get('momentum', 0.9),
                nesterov=optimizer_config.get('nesterov', False)
            )
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0),
                momentum=optimizer_config.get('momentum', 0),
                alpha=optimizer_config.get('alpha', 0.99)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build and return the learning rate scheduler."""
        if 'scheduler' not in self.config or self.config['scheduler'] is None:
            return None
        
        scheduler_config = self.config['scheduler']
        scheduler_name = scheduler_config['name'].lower()
        
        if scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'exponential':
            scheduler = ExponentialLR(
                optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 50),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_name == 'cosine_warm_restart':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 1),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return scheduler
    
    def build_criterion(self) -> nn.Module:
        """Build and return the loss criterion."""
        criterion_name = self.config.get('criterion', 'bce').lower()
        
        if criterion_name == 'bce':
            return nn.BCELoss()
        elif criterion_name == 'bce_with_logits':
            return nn.BCEWithLogitsLoss()
        elif criterion_name == 'focal':
            # Focal loss implementation that works with logits
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
                
                def forward(self, inputs, targets):
                    bce_loss = self.bce_with_logits(inputs, targets)
                    pt = torch.exp(-bce_loss)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                    return focal_loss.mean()
            
            return FocalLoss(
                alpha=self.config.get('focal_alpha', 1),
                gamma=self.config.get('focal_gamma', 2)
            )
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)
        for batch_idx, (drug_emb, prot_emb, labels) in enumerate(pbar):
            drug_emb = drug_emb.to(self.device)
            prot_emb = prot_emb.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(drug_emb, prot_emb)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if self.config.get('grad_clip_norm', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch metrics
            if self.config.get('use_wandb', False) and batch_idx % self.config.get('log_every', 100) == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for drug_emb, prot_emb, labels in tqdm(val_loader, desc="Validating", leave=False):
                drug_emb = drug_emb.to(self.device)
                prot_emb = prot_emb.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(drug_emb, prot_emb)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                
                # Calculate accuracy
                preds = (outputs > 0.5).float()
                total_correct += (preds == labels).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def should_early_stop(self, val_metric: float) -> bool:
        """Check if training should stop early."""
        if not self.config.get('early_stopping', {}).get('enabled', False):
            return False
        
        patience = self.config['early_stopping']['patience']
        monitor_mode = self.config.get('monitor_mode', 'min')
        
        if monitor_mode == 'min':
            improved = val_metric < self.best_val_metric
        else:
            improved = val_metric > self.best_val_metric
        
        if improved:
            self.best_val_metric = val_metric
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            return self.early_stop_counter >= patience
    
    def save_checkpoint(self, epoch: int, val_metric: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_metric': val_metric,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['val_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        # Get model dimensions from first batch
        sample_drug_emb, sample_prot_emb, _ = next(iter(train_loader))
        drug_dim = sample_drug_emb.shape[1]
        protein_dim = sample_prot_emb.shape[1]
        
        # Build model and training components
        self.model = self.build_model(drug_dim, protein_dim)
        self.optimizer = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(self.optimizer)
        self.criterion = self.build_criterion()
        
        # Load checkpoint if specified
        if self.config.get('resume_from_checkpoint'):
            self.load_checkpoint(self.config['resume_from_checkpoint'])
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch + 1, self.config['epochs'] + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Track with wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # Save checkpoint
            monitor_metric = val_loss if self.config.get('monitor_mode', 'min') == 'min' else val_acc
            is_best = (
                (self.config.get('monitor_mode', 'min') == 'min' and monitor_metric < self.best_val_metric) or
                (self.config.get('monitor_mode', 'min') == 'max' and monitor_metric > self.best_val_metric)
            )
            
            if is_best:
                self.best_val_metric = monitor_metric
            
            if epoch % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch, monitor_metric, is_best)
            
            # Early stopping
            if self.should_early_stop(monitor_metric):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_checkpoint(self.current_epoch, monitor_metric, False)
        
        if self.config.get('use_wandb', False):
            wandb.finish()
