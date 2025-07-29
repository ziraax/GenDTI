#!/usr/bin/env python3
"""
Standalone DTI Model Evaluation Script

This script provides comprehensive evaluation of trained DTI models with detailed metrics,
visualizations, and analysis capabilities.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import yaml
import json

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.utils.dti_dataset import DTIDataset
from training.evaluator import DTIEvaluator
from models.dti_model import DTIModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_test_dataloader(config: dict) -> DataLoader:
    """Create test dataloader."""
    test_dataset = DTIDataset(
        config['data']['test_triplet_path'],
        config['data']['drug_embeddings_path'],
        config['data']['protein_embeddings_path']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    return test_loader


def load_model(model_path: str, config: dict, device: torch.device) -> DTIModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        model_config = config['model']
    
    # Get model dimensions (we need to infer from data)
    # This is a limitation - we should save these in the checkpoint
    # For now, we'll use dummy values and update after seeing data
    model = DTIModel(
        drug_dim=1,  # Will be updated
        protein_dim=1,  # Will be updated
        fusion=model_config['fusion'],
        proj_dim=model_config['proj_dim'],
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout']
    )
    
    return model, checkpoint


def evaluate_model(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Use default configuration
        config = {
            'data': {
                'test_triplet_path': args.test_triplet_path,
                'drug_embeddings_path': args.drug_embeddings_path,
                'protein_embeddings_path': args.protein_embeddings_path
            },
            'batch_size': args.batch_size,
            'model': {
                'fusion': 'cross',
                'proj_dim': 256,
                'hidden_dim': 512,
                'dropout': 0.1
            }
        }
    
    # Create test dataloader
    test_loader = create_test_dataloader(config)
    
    # Get model dimensions from first batch
    sample_drug_emb, sample_prot_emb, _ = next(iter(test_loader))
    drug_dim = sample_drug_emb.shape[1]
    protein_dim = sample_prot_emb.shape[1]
    
    # Load model with correct dimensions
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        model_config = config['model']
    
    model = DTIModel(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        fusion=model_config['fusion'],
        proj_dim=model_config['proj_dim'],
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from: {args.model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create evaluator
    evaluator = DTIEvaluator(model, device)
    
    # Setup output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.model_path)
    
    print(f"Output directory: {output_dir}")
    
    # Perform basic evaluation
    print("\nPerforming basic evaluation...")
    basic_metrics = evaluator.evaluate(test_loader)
    evaluator.print_evaluation_summary(basic_metrics)
    
    # Perform detailed evaluation if requested
    if args.detailed:
        print("\nPerforming detailed evaluation...")
        detailed_results = evaluator.detailed_evaluation(test_loader, output_dir)
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_path}")
    
    # Perform threshold analysis if requested
    if args.threshold_analysis:
        print("\nPerforming threshold analysis...")
        import numpy as np
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_results = evaluator.threshold_analysis(test_loader, thresholds.tolist())
        
        # Save threshold analysis
        threshold_path = os.path.join(output_dir, 'threshold_analysis.csv')
        threshold_results.to_csv(threshold_path, index=False)
        
        print(f"Threshold analysis saved to: {threshold_path}")
        
        # Find optimal threshold
        optimal_idx = threshold_results['f1'].idxmax()
        optimal_threshold = threshold_results.loc[optimal_idx, 'threshold']
        optimal_f1 = threshold_results.loc[optimal_idx, 'f1']
        
        print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")
    
    print("\nEvaluation completed!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DTI model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to training configuration file (optional)"
    )
    parser.add_argument(
        "--test_triplet_path", 
        type=str, 
        default="data/processed/test.tsv",
        help="Path to test TSV file"
    )
    parser.add_argument(
        "--drug_embeddings_path", 
        type=str, 
        default="outputs/drug_embeddings/drug_embeddings.pt",
        help="Path to drug embeddings .pt file"
    )
    parser.add_argument(
        "--protein_embeddings_path", 
        type=str, 
        default="outputs/protein_embeddings/protein_embeddings.pt",
        help="Path to protein embeddings .pt file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save evaluation results (default: same as model directory)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Perform detailed evaluation with plots and analysis"
    )
    parser.add_argument(
        "--threshold_analysis", 
        action="store_true",
        help="Perform threshold analysis to find optimal decision threshold"
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
    
    evaluate_model(args)


if __name__ == "__main__":
    main()
