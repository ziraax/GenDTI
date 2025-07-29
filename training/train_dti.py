#!/usr/bin/env python3
"""
Advanced DTI Model Training Script

This script provides a comprehensive training pipeline for Drug-Target Interaction (DTI) models
with support for multiple optimizers, schedulers, loss functions, and advanced training features.
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import logging
from datetime import datetime

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.utils.dti_dataset import DTIDataset
from training.trainer import DTITrainer
from training.evaluator import DTIEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Create necessary directories for logging and saving."""
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)


def create_dataloaders(config: dict) -> tuple:
    """Create train, validation, and test dataloaders."""
    # Load datasets
    train_dataset = DTIDataset(
        config['data']['train_triplet_path'],
        config['data']['drug_embeddings_path'],
        config['data']['protein_embeddings_path']
    )
    
    val_dataset = DTIDataset(
        config['data']['val_triplet_path'],
        config['data']['drug_embeddings_path'],
        config['data']['protein_embeddings_path']
    )
    
    test_dataset = DTIDataset(
        config['data']['test_triplet_path'],
        config['data']['drug_embeddings_path'],
        config['data']['protein_embeddings_path']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def train_model(config: dict, train_loader: DataLoader, val_loader: DataLoader):
    """Train the DTI model."""
    # Initialize trainer
    trainer = DTITrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    return trainer


def evaluate_model(config: dict, test_loader: DataLoader, model_path: str = None):
    """Evaluate the trained model on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config['save_dir'], 'best_model.pt')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model dimensions from first batch
    sample_drug_emb, sample_prot_emb, _ = next(iter(test_loader))
    drug_dim = sample_drug_emb.shape[1]
    protein_dim = sample_prot_emb.shape[1]
    
    # Create model
    from models.dti_model import DTIModel
    model = DTIModel(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        fusion=config['model']['fusion'],
        proj_dim=config['model']['proj_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = DTIEvaluator(model, device)
    
    # Perform evaluation
    print("\nEvaluating model on test set...")
    test_results = evaluator.detailed_evaluation(
        test_loader, 
        save_dir=os.path.join(config['save_dir'], 'test_evaluation')
    )
    
    # Print summary
    evaluator.print_evaluation_summary(test_results['metrics'])
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Train DTI model with advanced features")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "both"],
        default="both",
        help="Training mode: train only, evaluate only, or both"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to trained model for evaluation (if mode is 'evaluate')"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default=None,
        help="Name for the experiment (overrides config)"
    )
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=None,
        help="GPU device ID to use"
    )
    
    args = parser.parse_args()
    
    # Set GPU device
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    # Generate experiment name if not provided
    if not config.get('experiment_name'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"dti_experiment_{timestamp}"
    
    # Update save directory to include experiment name
    config['save_dir'] = os.path.join(config['save_dir'], config['experiment_name'])
    config['log_dir'] = os.path.join(config['log_dir'], config['experiment_name'])
    
    # Setup directories
    setup_directories(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    print(f"Starting experiment: {config['experiment_name']}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Configuration: {args.config}")
    
    # Training
    if args.mode in ["train", "both"]:
        print("\n" + "="*50)
        print("Starting Training...")
        print("="*50)
        
        trainer = train_model(config, train_loader, val_loader)
        
        print("Training completed!")
    
    # Evaluation
    if args.mode in ["evaluate", "both"]:
        print("\n" + "="*50)
        print("Starting Evaluation...")
        print("="*50)
        
        model_path = args.model_path if args.model_path else None
        test_results = evaluate_model(config, test_loader, model_path)
        
        print("Evaluation completed!")
    
    print(f"\nExperiment '{config['experiment_name']}' finished!")
    print(f"Results saved in: {config['save_dir']}")


if __name__ == "__main__":
    main()
