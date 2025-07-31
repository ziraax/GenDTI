"""
RL fine-tuning script for conditional molecular generation.
Fine-tunes the diffusion model using REINFORCE with DTI + ADMET rewards.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np
import argparse
from typing import Dict, List, Optional
import wandb
from tqdm import tqdm

# Import your models
from models.graph_diffusion import ConditionalGraphDiffusion, graph_to_smiles
from models.reward import RewardFunction
from models.dti_model import DTIModel
from models.esm_encoder import ESMEncoder
from data.utils.dti_dataset import DTIDataset


class RLTrainer:
    """Trainer for RL fine-tuning of molecular generation."""
    
    def __init__(
        self,
        generator: ConditionalGraphDiffusion,
        dti_model: DTIModel,
        reward_function: RewardFunction,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict
    ):
        self.generator = generator
        self.dti_model = dti_model
        self.reward_function = reward_function
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Moving baseline for variance reduction
        self.baseline_ema = 0.0
        self.baseline_decay = config.get('baseline_decay', 0.99)
        
        # Statistics tracking
        self.step = 0
        self.total_rewards = []
        self.dti_scores = []
        self.admet_scores = []
        
    def train_step(
        self, 
        protein_embeddings: torch.Tensor,
        protein_sequences: List[str]
    ) -> Dict[str, float]:
        """
        Single training step of RL fine-tuning.
        
        Args:
            protein_embeddings: Protein embeddings [batch_size, protein_dim]
            protein_sequences: List of protein sequences for logging
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = protein_embeddings.size(0)
        num_samples = self.config.get('num_samples_per_protein', 8)
        
        # Generate molecules with gradients
        molecules, log_probs = self.generator.sample_for_rl(
            protein_embeddings=protein_embeddings,
            num_samples_per_protein=num_samples,
            temperature=self.config.get('temperature', 1.0),
            return_log_probs=True
        )
        
        # Convert to SMILES for reward computation
        smiles_list = []
        valid_molecules = []
        valid_log_probs = []
        
        for i, mol_data in enumerate(molecules.to_data_list()):
            smiles = graph_to_smiles(mol_data)
            if smiles and smiles != "C":  # Filter out trivial molecules
                smiles_list.append(smiles)
                valid_molecules.append(mol_data)
                valid_log_probs.append(log_probs[i])
            else:
                # Penalty for invalid molecules
                smiles_list.append("")
                valid_molecules.append(mol_data)
                valid_log_probs.append(log_probs[i])
        
        if len(valid_molecules) == 0:
            print("Warning: No valid molecules generated")
            return {'loss': 0.0, 'reward_mean': 0.0}
        
        # Expand protein embeddings to match generated molecules
        expanded_protein_embeddings = protein_embeddings.repeat_interleave(
            num_samples, dim=0
        )[:len(valid_molecules)]
        
        # Compute rewards
        reward_dict = self.reward_function.compute_reward(
            molecule_graphs=valid_molecules,
            protein_embeddings=expanded_protein_embeddings,
            smiles_list=smiles_list,
            normalize=True
        )
        
        total_rewards = reward_dict['total']
        dti_scores = reward_dict['dti']
        admet_scores = reward_dict['admet']
        
        # Update moving baseline
        current_reward_mean = total_rewards.mean().item()
        self.baseline_ema = (
            self.baseline_decay * self.baseline_ema + 
            (1 - self.baseline_decay) * current_reward_mean
        )
        
        # Compute baseline tensor
        baseline = torch.full_like(total_rewards, self.baseline_ema)
        
        # REINFORCE loss
        valid_log_probs_tensor = torch.stack(valid_log_probs)
        reinforce_loss = self.generator.reinforce_update(
            rewards=total_rewards,
            log_probs=valid_log_probs_tensor,
            baseline=baseline
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        reinforce_loss.backward()
        
        # Clear cache to save memory
        torch.cuda.empty_cache()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), 
            max_norm=self.config.get('grad_clip', 1.0)
        )
        
        self.optimizer.step()
        
        # Clear cache again after optimizer step
        torch.cuda.empty_cache()
        
        # Statistics
        self.step += 1
        self.total_rewards.extend(total_rewards.detach().cpu().numpy())
        self.dti_scores.extend(dti_scores.detach().cpu().numpy())
        self.admet_scores.extend(admet_scores.detach().cpu().numpy())
        
        # Return metrics
        return {
            'loss': reinforce_loss.item(),
            'reward_mean': current_reward_mean,
            'reward_std': total_rewards.std().item(),
            'dti_mean': dti_scores.mean().item(),
            'admet_mean': admet_scores.mean().item(),
            'baseline': self.baseline_ema,
            'valid_molecules': len(valid_molecules),
            'total_generated': len(molecules.to_data_list()),
            'success_rate': len([s for s in smiles_list if s and s != "C"]) / len(smiles_list)
        }
    
    def evaluate(
        self, 
        protein_embeddings: torch.Tensor,
        num_eval_samples: int = 32
    ) -> Dict[str, float]:
        """Evaluate the current model."""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate molecules
            molecules, _ = self.generator.sample_for_rl(
                protein_embeddings=protein_embeddings,
                num_samples_per_protein=num_eval_samples // len(protein_embeddings),
                temperature=0.8,  # Lower temperature for evaluation
                return_log_probs=False
            )
            
            # Convert to SMILES
            smiles_list = []
            valid_molecules = []
            
            for mol_data in molecules.to_data_list():
                smiles = graph_to_smiles(mol_data)
                smiles_list.append(smiles if smiles else "")
                if smiles and smiles != "C":
                    valid_molecules.append(mol_data)
            
            if len(valid_molecules) == 0:
                return {'eval_reward': 0.0, 'eval_validity': 0.0}
            
            # Expand protein embeddings
            expanded_protein_embeddings = protein_embeddings.repeat_interleave(
                num_eval_samples // len(protein_embeddings), dim=0
            )[:len(valid_molecules)]
            
            # Compute rewards
            reward_dict = self.reward_function.compute_reward(
                molecule_graphs=valid_molecules,
                protein_embeddings=expanded_protein_embeddings,
                smiles_list=[s for s in smiles_list if s and s != "C"],
                normalize=False  # Don't normalize for evaluation
            )
            
            validity = len(valid_molecules) / len(smiles_list)
            
            return {
                'eval_reward': reward_dict['total'].mean().item(),
                'eval_dti': reward_dict['dti'].mean().item(), 
                'eval_admet': reward_dict['admet'].mean().item(),
                'eval_validity': validity,
                'eval_samples': len(valid_molecules)
            }
        
        self.generator.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/rl_finetune.yaml')
    parser.add_argument('--generator_path', type=str, required=True)
    parser.add_argument('--dti_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/rl_finetuned')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='gendti-rl')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=config,
        name=f"rl_finetune_{config.get('beta_dti', 1.0)}_{config.get('beta_admet', 1.0)}"
    )
    
    # Load models
    print("Loading generator model...")
    generator = ConditionalGraphDiffusion(
        atom_feature_dim=7,
        edge_feature_dim=3,
        protein_dim=320,  # Match your protein embeddings
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 8),
        max_atoms=config.get('max_atoms', 50),
        timesteps=config.get('timesteps', 1000)
    ).to(device)
    
    # Load generator checkpoint
    generator_checkpoint = torch.load(args.generator_path, map_location=device)
    if 'model_state_dict' in generator_checkpoint:
        generator.load_state_dict(generator_checkpoint['model_state_dict'])
    else:
        generator.load_state_dict(generator_checkpoint)
    
    print("Loading DTI model...")
    # Load DTI checkpoint to get config
    dti_checkpoint = torch.load(args.dti_model_path, map_location=device)
    dti_config = dti_checkpoint['config']['model']
    
    dti_model = DTIModel(
        drug_dim=256,  # From GNN encoder output
        protein_dim=320,  # From your protein embeddings
        fusion=dti_config['fusion'],
        proj_dim=dti_config['proj_dim'],
        hidden_dim=dti_config['hidden_dim'],
        dropout=dti_config['dropout']
    ).to(device)
    
    dti_model.load_state_dict(dti_checkpoint['model_state_dict'])
    dti_model.eval()  # Keep DTI model frozen
    
    # Load GNN encoder for drug embeddings
    print("Loading GNN encoder...")
    from models.gnn_encoder import GNNEncoder
    gnn_checkpoint = torch.load('outputs/drug_embeddings/drug_embeddings_gnn_model.pt', map_location=device)
    gnn_config = gnn_checkpoint['config']
    
    gnn_encoder = GNNEncoder(
        input_dim=gnn_config['input_dim'],
        edge_dim=gnn_config['edge_dim'],
        hidden_dim=gnn_config['hidden_dim'],
        output_dim=gnn_config['output_dim'],
        num_layers=gnn_config['num_layers'],
        dropout=gnn_config['dropout']
    ).to(device)
    
    gnn_encoder.load_state_dict(gnn_checkpoint['model_state_dict'])
    gnn_encoder.eval()  # Keep GNN encoder frozen
    
    # Initialize reward function
    print("Initializing reward function...")
    reward_function = RewardFunction(
        dti_model=dti_model,
        drug_encoder=gnn_encoder,
        beta_dti=config.get('beta_dti', 1.0),
        beta_admet=config.get('beta_admet', 1.0),
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(
        generator.parameters(),
        lr=float(config.get('learning_rate', 1e-4)),
        weight_decay=float(config.get('weight_decay', 1e-5))
    )
    
    # Initialize trainer
    trainer = RLTrainer(
        generator=generator,
        dti_model=dti_model,
        reward_function=reward_function,
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = DTIDataset(
        triplet_path=f"{config.get('data_dir', 'data/processed')}/train.tsv",
        drug_embeddings_path="outputs/drug_embeddings/drug_embeddings.pt",
        protein_embeddings_path="outputs/protein_embeddings/protein_embeddings.pt"
    )
    
    # Limit dataset size for RL experiments!
    max_samples = config.get('max_samples', 100)
    if len(dataset) > max_samples:
        print(f"Limiting dataset from {len(dataset)} to {max_samples} samples for RL efficiency")
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=0
    )
    
    # Training loop
    print("Starting RL fine-tuning...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(config.get('num_epochs', 10)):
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Unpack the batch - DTIDataset returns (drug_emb, protein_emb, label)
            drug_embeddings, protein_embeddings, labels = batch
            protein_embeddings = protein_embeddings.to(device)
            
            # For RL, we don't have protein sequences, so create dummy ones for logging
            protein_sequences = [f"protein_{i}" for i in range(len(protein_embeddings))]
            
            # Training step
            metrics = trainer.train_step(protein_embeddings, protein_sequences)
            epoch_metrics.append(metrics)
            
            # Log to wandb
            wandb.log({
                'train/' + k: v for k, v in metrics.items()
            }, step=trainer.step)
            
            # Periodic evaluation
            if batch_idx % config.get('eval_interval', 100) == 0:
                eval_metrics = trainer.evaluate(protein_embeddings[:2])  # Small eval
                wandb.log({
                    'eval/' + k: v for k, v in eval_metrics.items()
                }, step=trainer.step)
        
        # Epoch summary
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }
        
        print(f"Epoch {epoch+1} - Reward: {avg_metrics['reward_mean']:.3f}, "
              f"Loss: {avg_metrics['loss']:.3f}, "
              f"Success Rate: {avg_metrics['success_rate']:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 5) == 0:
            checkpoint_path = os.path.join(args.output_dir, f'generator_rl_epoch_{epoch+1}.pt')
            torch.save(generator.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(args.output_dir, 'generator_rl_final.pt')
    torch.save(generator.state_dict(), final_path)
    print(f"Training complete. Final model saved: {final_path}")


if __name__ == "__main__":
    main()
