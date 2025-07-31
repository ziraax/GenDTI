#!/usr/bin/env python3
"""
Comprehensive testing script for the trained conditional diffusion model.
Tests the model with various protein sequences and evaluates generation quality.
"""

import torch
import numpy as np
import pandas as pd
import yaml
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up from tests/ to GenDTI/
sys.path.append(str(project_root))

from models.graph_diffusion import ConditionalGraphDiffusion
from models.diffusion_utils import GraphDiffusionProcess, DiffusionSchedule
from models.esm_encoder import ESMEncoder
from inference.generate_molecules import MolecularGenerator, MolecularEvaluator

class ModelTester:
    """Comprehensive tester for trained conditional diffusion model."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize the tester with model and config paths."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Using device: {self.device}")
        print(f"Loading model from: {model_path}")
        
        # Initialize components
        self._load_model()
        self._load_protein_encoder()
        self._load_test_proteins()
        
        # Initialize evaluator
        self.evaluator = MolecularEvaluator()
        
        print("Model tester initialized successfully!")
    
    def _load_model(self):
        """Load the trained diffusion model."""
        try:
            # Load model
            model_config = self.config['model']
            self.model = ConditionalGraphDiffusion(**model_config).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'loss' in checkpoint:
                    print(f"Model training loss: {checkpoint['loss']:.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
            
            self.model.eval()
            
            # Setup diffusion process
            diffusion_config = self.config.get('diffusion', {})
            schedule = DiffusionSchedule(**diffusion_config)
            self.diffusion = GraphDiffusionProcess(schedule)
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def _load_protein_encoder(self):
        """Load the protein encoder."""
        try:
            self.protein_encoder = ESMEncoder().to(self.device)
            self.protein_encoder.eval()
            print("✓ Protein encoder loaded successfully")
        except Exception as e:
            print(f"✗ Error loading protein encoder: {e}")
            raise
    
    def _load_test_proteins(self):
        """Load test protein sequences from data."""
        try:
            # Load test data
            test_data = pd.read_csv('data/processed/test.tsv', sep='\t')
            
            # Get unique proteins (sample a few for testing)
            unique_proteins = test_data['BindingDB_Target_Chain_Sequence'].unique()
            
            # Select diverse test proteins (different lengths)
            protein_lengths = [(seq, len(seq)) for seq in unique_proteins[:50]]
            protein_lengths.sort(key=lambda x: x[1])
            
            # Select proteins of different sizes
            self.test_proteins = []
            for i in range(0, len(protein_lengths), len(protein_lengths)//5):
                if i < len(protein_lengths):
                    self.test_proteins.append(protein_lengths[i][0])
            
            # Add some specific interesting proteins if available
            self.test_proteins = self.test_proteins[:5]  # Limit to 5 for testing
            
            print(f"✓ Loaded {len(self.test_proteins)} test proteins")
            for i, prot in enumerate(self.test_proteins):
                print(f"  Protein {i+1}: {len(prot)} amino acids")
                
        except Exception as e:
            print(f"✗ Error loading test proteins: {e}")
            # Fallback to dummy proteins
            self.test_proteins = [
                "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFKLI",
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
            ]
            print(f"✓ Using {len(self.test_proteins)} fallback proteins")
    
    @torch.no_grad()
    def test_basic_generation(self, num_molecules_per_protein: int = 5) -> Dict[str, Any]:
        """Test basic molecule generation functionality."""
        print(f"\n{'='*50}")
        print("BASIC GENERATION TEST")
        print(f"{'='*50}")
        
        results = {
            'test_type': 'basic_generation',
            'timestamp': datetime.now().isoformat(),
            'proteins_tested': len(self.test_proteins),
            'molecules_per_protein': num_molecules_per_protein,
            'results': []
        }
        
        for i, protein_seq in enumerate(self.test_proteins):
            print(f"\nTesting protein {i+1}/{len(self.test_proteins)} (length: {len(protein_seq)})")
            
            try:
                # Encode protein
                protein_emb = self.protein_encoder(protein_seq)
                if protein_emb.dim() == 1:
                    protein_emb = protein_emb.unsqueeze(0)
                
                print(f"  Protein embedding shape: {protein_emb.shape}")
                
                # Generate molecules
                print(f"  Generating {num_molecules_per_protein} molecules...")
                
                # Replicate protein embedding for multiple molecules
                protein_embs = protein_emb.repeat(num_molecules_per_protein, 1)
                
                # Use the model's improved sample_realistic method
                generated_batch = self.model.sample_realistic(
                    protein_embs, 
                    num_atoms=torch.tensor([20] * num_molecules_per_protein).to(self.device),  # Fixed size for consistent comparison across proteins
                    edge_threshold=0.3,  # Threshold for realistic connectivity
                    max_degree=4  # Maximum degree per atom for chemical realism
                )
                
                # Convert batch to individual molecules
                generated_graphs = generated_batch.to_data_list()
                
                print(f"  ✓ Generated {len(generated_graphs)} molecules")
                
                # Analyze generated molecules
                protein_result = {
                    'protein_idx': i,
                    'protein_length': len(protein_seq),
                    'molecules_generated': len(generated_graphs),
                    'generation_successful': True,
                    'molecules': []
                }
                
                for j, mol_graph in enumerate(generated_graphs):
                    mol_result = {
                        'molecule_idx': j,
                        'num_atoms': mol_graph.x.shape[0],
                        'num_edges': mol_graph.edge_index.shape[1],
                        'atom_features_shape': list(mol_graph.x.shape),
                        'edge_features_shape': list(mol_graph.edge_attr.shape) if hasattr(mol_graph, 'edge_attr') else None,
                        'has_coordinates': hasattr(mol_graph, 'pos'),
                        'coordinates_shape': list(mol_graph.pos.shape) if hasattr(mol_graph, 'pos') else None
                    }
                    protein_result['molecules'].append(mol_result)
                
                results['results'].append(protein_result)
                print(f"  ✓ Analysis complete")
                
            except Exception as e:
                print(f"  ✗ Error generating for protein {i+1}: {e}")
                protein_result = {
                    'protein_idx': i,
                    'protein_length': len(protein_seq),
                    'generation_successful': False,
                    'error': str(e)
                }
                results['results'].append(protein_result)
        
        return results
    
    @torch.no_grad()
    def test_molecular_properties(self, num_molecules: int = 10) -> Dict[str, Any]:
        """Test molecular property distribution of generated molecules."""
        print(f"\n{'='*50}")
        print("MOLECULAR PROPERTIES TEST")
        print(f"{'='*50}")
        
        results = {
            'test_type': 'molecular_properties',
            'timestamp': datetime.now().isoformat(),
            'total_molecules_generated': 0,
            'property_statistics': {}
        }
        
        all_molecules = []
        
        # Generate molecules from first protein
        protein_seq = self.test_proteins[0]
        protein_emb = self.protein_encoder(protein_seq)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)
        
        print(f"Generating {num_molecules} molecules for property analysis...")
        
        try:
            generated_batch = self.model.sample_realistic(
                protein_emb.repeat(num_molecules, 1),  # protein_emb is [1, 320], repeat to [num_molecules, 320]
                num_atoms=None,  # Let the model choose variable molecule sizes
                edge_threshold=0.3,  # Threshold for realistic connectivity
                max_degree=4  # Maximum degree per atom for chemical realism
            )
            
            generated_graphs = generated_batch.to_data_list()
            
            print(f"✓ Generated {len(generated_graphs)} molecules")
            
            # Analyze properties
            properties = {
                'num_atoms': [],
                'num_edges': [],
                'avg_degree': [],
                'feature_norms': []
            }
            
            for mol_graph in generated_graphs:
                properties['num_atoms'].append(mol_graph.x.shape[0])
                properties['num_edges'].append(mol_graph.edge_index.shape[1])
                
                # Calculate average degree
                num_nodes = mol_graph.x.shape[0]
                num_edges = mol_graph.edge_index.shape[1]
                avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
                properties['avg_degree'].append(avg_degree)
                
                # Feature statistics
                feature_norm = torch.norm(mol_graph.x, dim=1).mean().item()
                properties['feature_norms'].append(feature_norm)
            
            # Calculate statistics
            for prop_name, values in properties.items():
                if values:
                    results['property_statistics'][prop_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values[:10]  # First 10 values for inspection
                    }
            
            results['total_molecules_generated'] = len(generated_graphs)
            print(f"✓ Property analysis complete")
            
            # Print summary
            print(f"\nProperty Summary:")
            for prop_name, stats in results['property_statistics'].items():
                print(f"  {prop_name}: {stats['mean']:.2f} ± {stats['std']:.2f} (range: {stats['min']:.2f}-{stats['max']:.2f})")
                
        except Exception as e:
            print(f"✗ Error in property analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    @torch.no_grad()
    def test_conditioning_effect(self) -> Dict[str, Any]:
        """Test if different proteins lead to different generated molecules."""
        print(f"\n{'='*50}")
        print("PROTEIN CONDITIONING TEST")
        print(f"{'='*50}")
        
        results = {
            'test_type': 'conditioning_effect',
            'timestamp': datetime.now().isoformat(),
            'proteins_compared': min(3, len(self.test_proteins)),
            'molecules_per_protein': 5,
            'results': {}
        }
        
        # Test with first 3 proteins
        test_proteins = self.test_proteins[:3]
        protein_results = {}
        
        for i, protein_seq in enumerate(test_proteins):
            print(f"\nGenerating for protein {i+1} (length: {len(protein_seq)})...")
            
            try:
                protein_emb = self.protein_encoder(protein_seq)
                if protein_emb.dim() == 1:
                    protein_emb = protein_emb.unsqueeze(0)
                
                generated_batch = self.model.sample_realistic(
                    protein_emb.repeat(5, 1),
                    num_atoms=None,  # Let the model choose variable molecule sizes for better comparison
                    edge_threshold=0.3,  # Threshold for realistic connectivity
                    max_degree=4  # Maximum degree per atom for chemical realism
                )
                
                generated_graphs = generated_batch.to_data_list()
                
                # Collect features for comparison
                molecule_features = []
                for mol_graph in generated_graphs:
                    # Use mean atom features as molecule representation
                    mol_repr = mol_graph.x.mean(dim=0).cpu().numpy()
                    molecule_features.append(mol_repr)
                
                protein_results[f'protein_{i}'] = {
                    'protein_length': len(protein_seq),
                    'molecules_generated': len(generated_graphs),
                    'molecule_features': molecule_features,
                    'feature_mean': np.mean(molecule_features, axis=0).tolist(),
                    'feature_std': np.std(molecule_features, axis=0).tolist()
                }
                
                print(f"  ✓ Generated {len(generated_graphs)} molecules")
                
            except Exception as e:
                print(f"  ✗ Error with protein {i+1}: {e}")
                protein_results[f'protein_{i}'] = {'error': str(e)}
        
        # Compare protein differences
        if len(protein_results) >= 2:
            try:
                # Calculate pairwise distances between protein generations
                protein_keys = list(protein_results.keys())
                pairwise_distances = {}
                
                for i in range(len(protein_keys)):
                    for j in range(i+1, len(protein_keys)):
                        key1, key2 = protein_keys[i], protein_keys[j]
                        
                        if 'feature_mean' in protein_results[key1] and 'feature_mean' in protein_results[key2]:
                            mean1 = np.array(protein_results[key1]['feature_mean'])
                            mean2 = np.array(protein_results[key2]['feature_mean'])
                            
                            distance = np.linalg.norm(mean1 - mean2)
                            pairwise_distances[f'{key1}_vs_{key2}'] = float(distance)
                
                results['pairwise_distances'] = pairwise_distances
                results['mean_pairwise_distance'] = float(np.mean(list(pairwise_distances.values()))) if pairwise_distances else 0
                
                print(f"\nConditioning Effect Summary:")
                print(f"  Mean pairwise distance: {results['mean_pairwise_distance']:.4f}")
                for pair, dist in pairwise_distances.items():
                    print(f"  {pair}: {dist:.4f}")
                    
            except Exception as e:
                print(f"Error in conditioning analysis: {e}")
                results['conditioning_analysis_error'] = str(e)
        
        results['results'] = protein_results
        return results
    
    def test_model_stability(self, num_runs: int = 3) -> Dict[str, Any]:
        """Test model stability across multiple runs."""
        print(f"\n{'='*50}")
        print("MODEL STABILITY TEST")
        print(f"{'='*50}")
        
        results = {
            'test_type': 'model_stability',
            'timestamp': datetime.now().isoformat(),
            'num_runs': num_runs,
            'runs': []
        }
        
        protein_seq = self.test_proteins[0]
        protein_emb = self.protein_encoder(protein_seq)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)
        
        print(f"Testing stability across {num_runs} runs...")
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            
            try:
                generated_batch = self.model.sample_realistic(
                    protein_emb.repeat(3, 1),
                    num_atoms=None,  # Let the model choose variable molecule sizes for stability testing
                    edge_threshold=0.3,  # Threshold for realistic connectivity
                    max_degree=4  # Maximum degree per atom for chemical realism
                )
                
                generated_graphs = generated_batch.to_data_list()
                
                run_result = {
                    'run_idx': run,
                    'molecules_generated': len(generated_graphs),
                    'molecule_sizes': [g.x.shape[0] for g in generated_graphs],
                    'feature_means': [g.x.mean().item() for g in generated_graphs],
                    'feature_stds': [g.x.std().item() for g in generated_graphs]
                }
                
                results['runs'].append(run_result)
                
            except Exception as e:
                print(f"    ✗ Error in run {run + 1}: {e}")
                results['runs'].append({'run_idx': run, 'error': str(e)})
        
        # Analyze stability
        successful_runs = [r for r in results['runs'] if 'error' not in r]
        if len(successful_runs) >= 2:
            all_means = []
            all_stds = []
            
            for run in successful_runs:
                all_means.extend(run['feature_means'])
                all_stds.extend(run['feature_stds'])
            
            results['stability_metrics'] = {
                'mean_consistency': float(np.std(all_means)),
                'std_consistency': float(np.std(all_stds)),
                'successful_runs': len(successful_runs),
                'total_runs': num_runs
            }
            
            print(f"✓ Stability analysis complete")
            print(f"  Mean consistency (lower is better): {results['stability_metrics']['mean_consistency']:.4f}")
            print(f"  Std consistency (lower is better): {results['stability_metrics']['std_consistency']:.4f}")
        
        return results
    
    def run_comprehensive_test(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run all tests and compile results."""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE MODEL TESTING")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Test proteins: {len(self.test_proteins)}")
        
        all_results = {
            'model_path': self.model_path,
            'device': str(self.device),
            'test_timestamp': datetime.now().isoformat(),
            'model_config': self.config,
            'tests': {}
        }
        
        # Run all tests
        try:
            all_results['tests']['basic_generation'] = self.test_basic_generation()
        except Exception as e:
            print(f"Basic generation test failed: {e}")
            all_results['tests']['basic_generation'] = {'error': str(e)}
        
        try:
            all_results['tests']['molecular_properties'] = self.test_molecular_properties()
        except Exception as e:
            print(f"Molecular properties test failed: {e}")
            all_results['tests']['molecular_properties'] = {'error': str(e)}
        
        try:
            all_results['tests']['conditioning_effect'] = self.test_conditioning_effect()
        except Exception as e:
            print(f"Conditioning effect test failed: {e}")
            all_results['tests']['conditioning_effect'] = {'error': str(e)}
        
        try:
            all_results['tests']['model_stability'] = self.test_model_stability()
        except Exception as e:
            print(f"Model stability test failed: {e}")
            all_results['tests']['model_stability'] = {'error': str(e)}
        
        # Generate summary
        self._generate_test_summary(all_results)
        
        # Save results
        if output_file:
            # Convert any numpy arrays to lists for JSON serialization
            json_results = self._make_json_serializable(all_results)
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\n✓ Detailed results saved to: {output_file}")
        
        return all_results
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_test_summary(self, results: Dict[str, Any]):
        """Generate and print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        for test_name, test_results in results['tests'].items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            
            if 'error' in test_results:
                print(f"  ✗ FAILED: {test_results['error']}")
                continue
            
            if test_name == 'basic_generation':
                successful = sum(1 for r in test_results['results'] if r.get('generation_successful', False))
                total = len(test_results['results'])
                print(f"  ✓ {successful}/{total} proteins generated successfully")
                
                if successful > 0:
                    total_molecules = sum(r.get('molecules_generated', 0) for r in test_results['results'] if r.get('generation_successful'))
                    print(f"  ✓ Total molecules generated: {total_molecules}")
            
            elif test_name == 'molecular_properties':
                if 'total_molecules_generated' in test_results:
                    print(f"  ✓ Analyzed {test_results['total_molecules_generated']} molecules")
                    if 'property_statistics' in test_results:
                        stats = test_results['property_statistics']
                        if 'num_atoms' in stats:
                            print(f"  ✓ Avg atoms per molecule: {stats['num_atoms']['mean']:.1f} ± {stats['num_atoms']['std']:.1f}")
                        if 'avg_degree' in stats:
                            print(f"  ✓ Avg node degree: {stats['avg_degree']['mean']:.2f} ± {stats['avg_degree']['std']:.2f}")
            
            elif test_name == 'conditioning_effect':
                if 'mean_pairwise_distance' in test_results:
                    print(f"  ✓ Mean pairwise distance: {test_results['mean_pairwise_distance']:.4f}")
                    if test_results['mean_pairwise_distance'] > 0.1:
                        print(f"  ✓ Good conditioning effect detected")
                    else:
                        print(f"  ⚠ Weak conditioning effect")
            
            elif test_name == 'model_stability':
                if 'stability_metrics' in test_results:
                    metrics = test_results['stability_metrics']
                    print(f"  ✓ Successful runs: {metrics['successful_runs']}/{metrics['total_runs']}")
                    print(f"  ✓ Mean consistency: {metrics['mean_consistency']:.4f}")
        
        print(f"\n{'='*60}")
        print("Testing complete!")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test trained conditional diffusion model')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', default='configs/diffusion_conditional.yaml', help='Path to model config')
    parser.add_argument('--output', help='Output file for detailed results')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Initialize tester
    tester = ModelTester(args.model, args.config)
    
    # Run tests
    if args.quick:
        print("Running quick tests...")
        results = tester.test_basic_generation(num_molecules_per_protein=2)
    else:
        print("Running comprehensive tests...")
        results = tester.run_comprehensive_test(args.output)
    
    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main()
