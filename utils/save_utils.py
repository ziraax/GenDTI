import torch
import os 
import json
from pathlib import Path
from typing import Dict, Any


class SaveManager:
    """
    Manages saving and loading of model checkpoints and training state.
    """
    
    def __init__(self, save_dir: str, experiment_name: str = "experiment"):
        """
        Args:
            save_dir: Directory to save checkpoints
            experiment_name: Name of the experiment (used for subdirectories)
        """
        if experiment_name:
            self.save_dir = Path(save_dir) / experiment_name
        else:
            self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or "experiment"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        loss: float,
        metrics: Dict[str, float] = None,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            loss: Current loss
            metrics: Additional metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics or {}
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = self.save_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save as best if needed
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        checkpoint_path: str = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Path to checkpoint, defaults to latest
            
        Returns:
            Dictionary with epoch, loss, and metrics
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / 'latest.pt'
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'metrics': checkpoint.get('metrics', {})
        }


def save_embeddings(embeddings, id_list, output_path):
    assert len(embeddings) == len(id_list)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    torch.save({
        "embeddings": embeddings,
        "ids": id_list
    }, output_path)


def save_mapping(sequences_ids, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    seq_to_idx = {seq: i for i, seq in enumerate(sequences_ids)}
    map_path = os.path.splitext(output_path)[0] + "_seq_to_idx.json"
    with open(map_path, 'w') as f:
        json.dump(seq_to_idx, f)
    print(f"Saved sequence to index mapping to {map_path}")