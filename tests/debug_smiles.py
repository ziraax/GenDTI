#!/usr/bin/env python3
"""
Debug SMILES processing issues.
"""

import sys
import os
sys.path.append('/home/huwalter/WorkingFolderHugoWALTER/GenDTI')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available!")
    RDKIT_AVAILABLE = False

def debug_smiles_processing():
    """Debug individual SMILES processing."""
    
    test_smiles = ["CC", "C=C", "C#C", "C#N", "CCO"]
    
    for smiles in test_smiles:
        print(f"\nTesting: {smiles}")
        
        if not RDKIT_AVAILABLE:
            print("  RDKit not available")
            continue
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("  ❌ Failed to parse SMILES")
                continue
                
            print(f"  ✅ Parsed successfully")
            
            # Remove hydrogens
            mol_no_h = Chem.RemoveHs(mol)
            num_atoms = mol_no_h.GetNumAtoms()
            num_bonds = mol_no_h.GetNumBonds()
            
            print(f"  Atoms: {num_atoms}")
            print(f"  Bonds: {num_bonds}")
            
            # Check filtering conditions
            if num_atoms > 50:
                print("  ❌ Too many atoms (> 50)")
            elif num_atoms < 3:
                print("  ❌ Too few atoms (< 3)")
            else:
                print("  ✅ Atom count OK")
                
            # Show atom details
            print("  Atom details:")
            for i, atom in enumerate(mol_no_h.GetAtoms()):
                print(f"    {i}: {atom.GetSymbol()}")
                
            # Show bond details
            print("  Bond details:")
            for i, bond in enumerate(mol_no_h.GetBonds()):
                print(f"    {i}: {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} ({bond.GetBondType()})")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    debug_smiles_processing()
