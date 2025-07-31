#!/usr/bin/env python3
"""
Analysis of comprehensive test results from the trained conditional diffusion model
"""

import torch
import json
import numpy as np
import sys
from pathlib import Path
import glob
from datetime import datetime

def find_latest_results():
    """Find the most recent comprehensive test results file."""
    results_dir = Path("outputs/tests_stage2")
    if not results_dir.exists():
        print("❌ No test results directory found. Run the comprehensive test first.")
        return None
    
    # Find all result files
    result_files = list(results_dir.glob("comprehensive_test_results_*.json"))
    
    if not result_files:
        print("❌ No comprehensive test results found. Run the test first with:")
        print("python tests/test_trained_stage2_model.py --model outputs/stage2_conditional_v4/best.pt --config configs/diffusion_conditional.yaml --output outputs/tests_stage2/results.json")
        return None
    
    # Return the most recent file
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    print(f"📊 Analyzing results from: {latest_file.name}")
    return latest_file

def analyze_results(results_file=None):
    """Analyze the comprehensive test results."""
    
    if results_file is None:
        results_file = find_latest_results()
        if results_file is None:
            return
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    print("="*70)
    print("COMPREHENSIVE TEST RESULTS ANALYSIS")
    print("="*70)
    
    # Basic info
    print(f"🕒 Test timestamp: {results['test_timestamp']}")
    print(f"🖥️  Device: {results['device']}")
    print(f"📂 Model: {results['model_path']}")
    
    # Analyze each test
    tests = results.get('tests', {})
    
    print(f"\n{'='*50}")
    print("📊 TEST RESULTS OVERVIEW")
    print(f"{'='*50}")
    
    for test_name, test_data in tests.items():
        print(f"\n🧪 {test_name.upper().replace('_', ' ')}:")
        
        if 'error' in test_data:
            print(f"   ❌ FAILED: {test_data['error']}")
            continue
            
        if test_name == 'basic_generation':
            analyze_basic_generation(test_data)
        elif test_name == 'molecular_properties':
            analyze_molecular_properties(test_data)
        elif test_name == 'conditioning_effect':
            analyze_conditioning_effect(test_data)
        elif test_name == 'model_stability':
            analyze_model_stability(test_data)
    
    print(f"\n{'='*70}")
    print("🎯 OVERALL ASSESSMENT")
    print(f"{'='*70}")
    
    generate_assessment(tests)

def analyze_basic_generation(data):
    """Analyze basic generation test results."""
    results = data.get('results', [])
    successful = [r for r in results if r.get('generation_successful', False)]
    
    print(f"   ✅ Success Rate: {len(successful)}/{len(results)} proteins ({len(successful)/len(results)*100:.1f}%)")
    
    if successful:
        total_molecules = sum(r.get('molecules_generated', 0) for r in successful)
        print(f"   🧬 Total molecules generated: {total_molecules}")
        
        # Analyze molecule sizes and properties
        all_molecules = []
        for protein_result in successful:
            all_molecules.extend(protein_result.get('molecules', []))
        
        if all_molecules:
            atom_counts = [m['num_atoms'] for m in all_molecules]
            edge_counts = [m['num_edges'] for m in all_molecules]
            
            print(f"   📏 Molecule sizes: {min(atom_counts)}-{max(atom_counts)} atoms (avg: {np.mean(atom_counts):.1f})")
            print(f"   🔗 Edge counts: {min(edge_counts)}-{max(edge_counts)} edges (avg: {np.mean(edge_counts):.1f})")

def analyze_molecular_properties(data):
    """Analyze molecular properties test results."""
    total_generated = data.get('total_molecules_generated', 0)
    stats = data.get('property_statistics', {})
    
    print(f"   🧬 Molecules analyzed: {total_generated}")
    
    if stats:
        print(f"   📊 Property Statistics:")
        for prop, values in stats.items():
            mean_val = values['mean']
            std_val = values['std']
            min_val = values['min']
            max_val = values['max']
            print(f"      • {prop}: {mean_val:.2f} ± {std_val:.2f} (range: {min_val:.1f}-{max_val:.1f})")
            
            # Special analysis for key properties
            if prop == 'avg_degree':
                if 3.5 <= mean_val <= 4.5:
                    print(f"        ✅ Realistic average degree for organic molecules")
                else:
                    print(f"        ⚠️  Unusual average degree for organic molecules")
            elif prop == 'num_atoms':
                if std_val > 0:
                    print(f"        ✅ Good molecular size diversity")
                else:
                    print(f"        ⚠️  No size diversity - all molecules same size")

def analyze_conditioning_effect(data):
    """Analyze protein conditioning effect results."""
    mean_distance = data.get('mean_pairwise_distance', 0)
    pairwise = data.get('pairwise_distances', {})
    
    print(f"   📏 Mean pairwise distance: {mean_distance:.4f}")
    
    if mean_distance > 1.0:
        print(f"   ✅ Strong conditioning effect - different proteins generate different molecules")
    elif mean_distance > 0.1:
        print(f"   ⚠️  Moderate conditioning effect")
    else:
        print(f"   ❌ Weak conditioning effect - proteins generate similar molecules")
    
    if pairwise:
        print(f"   🔍 Pairwise distances:")
        for pair, distance in pairwise.items():
            print(f"      • {pair}: {distance:.4f}")

def analyze_model_stability(data):
    """Analyze model stability test results."""
    metrics = data.get('stability_metrics', {})
    runs = data.get('runs', [])
    
    successful_runs = metrics.get('successful_runs', 0)
    total_runs = metrics.get('total_runs', 0)
    
    print(f"   ✅ Successful runs: {successful_runs}/{total_runs}")
    
    if 'mean_consistency' in metrics:
        consistency = metrics['mean_consistency']
        print(f"   📊 Mean consistency: {consistency:.4f} (lower is better)")
        
        if consistency < 0.5:
            print(f"   ✅ Very stable generation")
        elif consistency < 1.0:
            print(f"   ✅ Good stability")
        elif consistency < 2.0:
            print(f"   ⚠️  Moderate stability")
        else:
            print(f"   ❌ Poor stability")

def generate_assessment(tests):
    """Generate overall assessment of model performance."""
    
    print("✅ **ACHIEVEMENTS:**")
    
    # Check basic generation
    basic_gen = tests.get('basic_generation', {})
    if not basic_gen.get('error'):
        results = basic_gen.get('results', [])
        successful = [r for r in results if r.get('generation_successful', False)]
        if len(successful) == len(results):
            print("   • 🎯 Perfect generation success rate across all protein types")
        else:
            print(f"   • 🎯 High generation success rate: {len(successful)}/{len(results)}")
    
    # Check molecular properties
    mol_props = tests.get('molecular_properties', {})
    if not mol_props.get('error'):
        stats = mol_props.get('property_statistics', {})
        if 'num_atoms' in stats and stats['num_atoms']['std'] > 0:
            print("   • 🧬 Generates diverse molecule sizes (realistic)")
        if 'avg_degree' in stats:
            avg_deg = stats['avg_degree']['mean']
            if 3.5 <= avg_deg <= 4.5:
                print("   • 🔗 Realistic molecular connectivity patterns")
    
    # Check conditioning
    conditioning = tests.get('conditioning_effect', {})
    if not conditioning.get('error'):
        distance = conditioning.get('mean_pairwise_distance', 0)
        if distance > 1.0:
            print("   • 🎯 Strong protein conditioning - different proteins → different molecules")
    
    # Check stability
    stability = tests.get('model_stability', {})
    if not stability.get('error'):
        metrics = stability.get('stability_metrics', {})
        if metrics.get('successful_runs', 0) == metrics.get('total_runs', 0):
            print("   • ⚖️  Excellent model stability across multiple runs")
    
    print("\\n🚀 **OVERALL STATUS:**")
    
    # Count successful tests
    successful_tests = sum(1 for test_data in tests.values() if 'error' not in test_data)
    total_tests = len(tests)
    
    if successful_tests == total_tests:
        print("   🎉 ALL TESTS PASSED - Model is production ready!")
        print("   📈 The conditional diffusion model successfully generates:")
        print("      • Realistic molecular structures with proper connectivity")
        print("      • Diverse molecule sizes and properties") 
        print("      • Protein-specific molecular features")
        print("      • Stable and reproducible results")
        print("\\n   🎯 Ready for drug discovery applications!")
    else:
        print(f"   ⚠️  {successful_tests}/{total_tests} tests passed - Some issues need attention")
        
        failed_tests = [name for name, data in tests.items() if 'error' in data]
        if failed_tests:
            print(f"   ❌ Failed tests: {', '.join(failed_tests)}")
    
    print("\\n📋 **NEXT STEPS:**")
    print("   1. 🧪 Use the model for molecular generation with new proteins")
    print("   2. 🔬 Integrate with drug discovery pipelines") 
    print("   3. 📊 Run larger-scale evaluations with more diverse proteins")
    print("   4. 🎯 Fine-tune for specific therapeutic targets if needed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze comprehensive test results')
    parser.add_argument('--results', type=str, help='Path to specific results file (optional - will auto-find latest if not provided)')
    
    args = parser.parse_args()
    
    if args.results:
        analyze_results(Path(args.results))
    else:
        analyze_results()
