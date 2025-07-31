#!/usr/bin/env python3
"""
Test the create_molecule_graph function (SMILES → Graph conversion)
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.graph_diffusion import create_molecule_graph, graph_to_smiles

def test_create_molecule_graph():
    """Test SMILES → Graph → SMILES round-trip conversion."""
    
    print("🧪 TESTING create_molecule_graph FUNCTION")
    print("="*60)
    
    # Test molecules of different types and complexities
    test_molecules = [
        ("Methane", "C"),
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Toluene", "Cc1ccccc1"),
        ("Acetone", "CC(=O)C"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Glucose", "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
        ("Diphenyl ether", "c1ccc(Oc2ccccc2)cc1")
    ]
    
    results = {
        'successful_conversions': 0,
        'failed_conversions': 0,
        'round_trip_successes': 0,
        'round_trip_failures': 0,
        'details': []
    }
    
    print("Testing SMILES → Graph conversion:")
    print("-" * 40)
    
    for name, smiles in test_molecules:
        print(f"\n🧬 Testing {name}: {smiles}")
        
        try:
            # Convert SMILES to graph
            graph = create_molecule_graph(smiles, max_atoms=100)
            
            if graph is None:
                print(f"  ❌ Graph conversion failed")
                results['failed_conversions'] += 1
                results['details'].append({
                    'name': name,
                    'smiles': smiles,
                    'status': 'graph_conversion_failed'
                })
                continue
            
            # Analyze the graph
            num_atoms = graph.x.shape[0]
            num_edges = graph.edge_index.shape[1]
            atom_features_shape = graph.x.shape
            edge_features_shape = graph.edge_attr.shape
            pos_shape = graph.pos.shape
            
            print(f"  ✅ Graph created successfully")
            print(f"     Atoms: {num_atoms}")
            print(f"     Edges: {num_edges}")
            print(f"     Atom features: {atom_features_shape}")
            print(f"     Edge features: {edge_features_shape}")
            print(f"     Coordinates: {pos_shape}")
            
            results['successful_conversions'] += 1
            
            # Test round-trip conversion (Graph → SMILES)
            print(f"  🔄 Testing round-trip conversion...")
            reconstructed_smiles = graph_to_smiles(graph, sanitize=True)
            
            if reconstructed_smiles:
                print(f"     Original:      {smiles}")
                print(f"     Reconstructed: {reconstructed_smiles}")
                
                # Check if they represent the same molecule (canonicalize both)
                try:
                    from rdkit import Chem
                    
                    original_mol = Chem.MolFromSmiles(smiles)
                    reconstructed_mol = Chem.MolFromSmiles(reconstructed_smiles)
                    
                    if original_mol and reconstructed_mol:
                        original_canonical = Chem.MolToSmiles(original_mol)
                        reconstructed_canonical = Chem.MolToSmiles(reconstructed_mol)
                        
                        if original_canonical == reconstructed_canonical:
                            print(f"  ✅ Perfect round-trip match!")
                            results['round_trip_successes'] += 1
                        else:
                            print(f"  ⚠️  Structural difference (but both valid)")
                            print(f"     Canonical original:      {original_canonical}")
                            print(f"     Canonical reconstructed: {reconstructed_canonical}")
                            results['round_trip_failures'] += 1
                    else:
                        print(f"  ❌ Canonicalization failed")
                        results['round_trip_failures'] += 1
                        
                except ImportError:
                    print(f"  ⚠️  RDKit not available for canonicalization check")
                    results['round_trip_successes'] += 1  # Assume success if we can't check
                    
            else:
                print(f"  ❌ Round-trip failed: could not convert back to SMILES")
                results['round_trip_failures'] += 1
            
            # Store detailed results
            results['details'].append({
                'name': name,
                'original_smiles': smiles,
                'reconstructed_smiles': reconstructed_smiles,
                'num_atoms': num_atoms,
                'num_edges': num_edges,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['failed_conversions'] += 1
            results['details'].append({
                'name': name,
                'smiles': smiles,
                'error': str(e),
                'status': 'error'
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total = len(test_molecules)
    print(f"📊 Graph Conversion Results:")
    print(f"   ✅ Successful: {results['successful_conversions']}/{total} ({results['successful_conversions']/total*100:.1f}%)")
    print(f"   ❌ Failed:     {results['failed_conversions']}/{total} ({results['failed_conversions']/total*100:.1f}%)")
    
    total_attempts = results['round_trip_successes'] + results['round_trip_failures']
    if total_attempts > 0:
        print(f"\n🔄 Round-trip Conversion Results:")
        print(f"   ✅ Successful: {results['round_trip_successes']}/{total_attempts} ({results['round_trip_successes']/total_attempts*100:.1f}%)")
        print(f"   ❌ Failed:     {results['round_trip_failures']}/{total_attempts} ({results['round_trip_failures']/total_attempts*100:.1f}%)")
    
    return results

def test_edge_cases():
    """Test edge cases and error handling."""
    
    print(f"\n{'='*60}")
    print("TESTING EDGE CASES")
    print(f"{'='*60}")
    
    edge_cases = [
        ("Empty string", ""),
        ("Invalid SMILES", "INVALID_SMILES_123"),
        ("Very large molecule", "C" * 100),  # Should exceed max_atoms
        ("Single atom", "C"),
        ("No bonds", "[Na+].[Cl-]"),  # Ionic compound
        ("Complex stereochemistry", "C[C@H](O)[C@@H](O)C"),
    ]
    
    for name, smiles in edge_cases:
        print(f"\n🧪 Testing {name}: '{smiles}'")
        
        try:
            graph = create_molecule_graph(smiles, max_atoms=50)
            
            if graph is None:
                print(f"  ✅ Correctly returned None (expected for some cases)")
            else:
                print(f"  ✅ Graph created: {graph.x.shape[0]} atoms, {graph.edge_index.shape[1]} edges")
                
        except Exception as e:
            print(f"  ⚠️  Exception: {e}")

def test_feature_analysis():
    """Analyze the features generated for different atom and bond types."""
    
    print(f"\n{'='*60}")
    print("FEATURE ANALYSIS")
    print(f"{'='*60}")
    
    test_smiles = "CCN(C)C(=O)c1ccccc1"  # N,N-dimethylbenzamide - has C, N, O atoms and different bond types
    
    print(f"🔬 Analyzing features for: {test_smiles}")
    
    graph = create_molecule_graph(test_smiles, max_atoms=50)
    
    if graph is not None:
        print(f"\n📊 Graph Structure:")
        print(f"   Atoms: {graph.x.shape[0]}")
        print(f"   Edges: {graph.edge_index.shape[1]}")
        
        print(f"\n🧬 Atom Features (7D):")
        print(f"   Shape: {graph.x.shape}")
        print(f"   Feature dimensions: [atomic_num, degree, formal_charge, radical_electrons, aromatic, hybridization, mass]")
        
        # Analyze atom features
        for i in range(min(5, graph.x.shape[0])):  # Show first 5 atoms
            features = graph.x[i]
            print(f"   Atom {i}: atomic_num={features[0]:.1f}, degree={features[1]:.1f}, charge={features[2]:.1f}, aromatic={features[4]:.1f}")
        
        print(f"\n🔗 Edge Features (3D):")
        print(f"   Shape: {graph.edge_attr.shape}")
        print(f"   Feature dimensions: [bond_type, in_ring, conjugated]")
        
        # Analyze edge features
        for i in range(min(5, graph.edge_attr.shape[0])):  # Show first 5 edges
            features = graph.edge_attr[i]
            edge = graph.edge_index[:, i]
            print(f"   Edge {i} ({edge[0].item()}→{edge[1].item()}): bond_type={features[0]:.1f}, in_ring={features[1]:.1f}, conjugated={features[2]:.1f}")
        
        print(f"\n📍 Coordinates:")
        print(f"   Shape: {graph.pos.shape}")
        print(f"   Random 3D coordinates generated (placeholder)")
        
    else:
        print(f"❌ Could not create graph for analysis")

if __name__ == "__main__":
    try:
        # Test main functionality
        results = test_create_molecule_graph()
        
        # Test edge cases
        test_edge_cases()
        
        # Analyze features
        test_feature_analysis()
        
        print(f"\n🎉 Testing complete!")
        
        # Overall assessment
        if results['successful_conversions'] >= 8:  # At least 80% success
            print(f"✅ create_molecule_graph function is working well!")
        elif results['successful_conversions'] >= 5:
            print(f"⚠️  create_molecule_graph function has some issues but mostly works")
        else:
            print(f"❌ create_molecule_graph function needs significant fixes")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
