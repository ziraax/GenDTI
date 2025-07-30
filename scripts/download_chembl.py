"""
Download and preprocess ChEMBL molecular data for diffusion model training.
"""

import os
import sys
import requests
import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import gzip
import urllib.request
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: pip install rdkit")
    RDKIT_AVAILABLE = False


class ChEMBLDownloader:
    """
    Download and preprocess ChEMBL molecular database.
    """
    
    def __init__(self, data_dir: str = "data/raw/chembl"):
        """
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # ChEMBL download URLs (adjust version as needed)
        self.chembl_version = "34"
        self.base_url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{self.chembl_version}"
        
        self.files_to_download = {
            'chembl_sqlite': f'chembl_{self.chembl_version}_sqlite.tar.gz',
            # SDF file is optional - we can get molecules from SQLite
            # 'molecules_sdf': f'chembl_{self.chembl_version}_molecules.sdf.gz'
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def download_chembl_data(self, force_redownload: bool = False) -> bool:
        """Download ChEMBL database files."""
        print(f"Downloading ChEMBL {self.chembl_version} data...")
        
        critical_files = ['chembl_sqlite']  # Files we absolutely need
        optional_files = ['molecules_sdf']  # Files that are nice to have but not required
        
        success = True
        for file_type, filename in self.files_to_download.items():
            filepath = self.data_dir / filename
            
            if filepath.exists() and not force_redownload:
                print(f"File {filename} already exists, skipping download")
                continue
            
            url = f"{self.base_url}/{filename}"
            print(f"Downloading {filename}...")
            
            if not self.download_file(url, filepath):
                print(f"Failed to download {filename}")
                if file_type in critical_files:
                    print(f"Critical file {filename} failed to download!")
                    success = False
                else:
                    print(f"Optional file {filename} failed - continuing anyway")
            else:
                print(f"Successfully downloaded {filename}")
        
        return success
    
    def extract_files(self) -> bool:
        """Extract downloaded files."""
        import tarfile
        
        print("Extracting files...")
        
        # Extract SQLite database
        sqlite_archive = self.data_dir / self.files_to_download['chembl_sqlite']
        if sqlite_archive.exists():
            print("Extracting SQLite database...")
            with tarfile.open(sqlite_archive, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            print("SQLite database extracted")
        
        return True
    
    def get_sqlite_path(self) -> Optional[Path]:
        """Get path to extracted SQLite database."""
        # Look for .db file in the data directory (recursive search)
        for db_file in self.data_dir.rglob("*.db"):
            if db_file.is_file():
                return db_file
        
        # Alternative direct paths
        possible_paths = [
            self.data_dir / f"chembl_{self.chembl_version}.db",
            self.data_dir / f"chembl_{self.chembl_version}" / f"chembl_{self.chembl_version}.db",
            self.data_dir / f"chembl_{self.chembl_version}" / f"chembl_{self.chembl_version}_sqlite" / f"chembl_{self.chembl_version}.db"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
            
        return None


class ChEMBLProcessor:
    """
    Process ChEMBL data for molecular diffusion training.
    """
    
    def __init__(self, sqlite_path: str, output_dir: str = "data/processed/chembl"):
        """
        Args:
            sqlite_path: Path to ChEMBL SQLite database
            output_dir: Directory to save processed data
        """
        self.sqlite_path = Path(sqlite_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular processing")
    
    def extract_molecules(
        self,
        max_molecules: Optional[int] = None,
        min_atoms: int = 5,
        max_atoms: int = 50,
        filter_organics: bool = True
    ) -> pd.DataFrame:
        """
        Extract molecules from ChEMBL database.
        
        Args:
            max_molecules: Maximum number of molecules to extract
            min_atoms: Minimum number of atoms
            max_atoms: Maximum number of atoms
            filter_organics: Only include organic molecules
            
        Returns:
            DataFrame with molecule data
        """
        print("Connecting to ChEMBL database...")
        conn = sqlite3.connect(self.sqlite_path)
        
        # Query to get molecules with SMILES
        query = """
        SELECT 
            cs.molregno,
            cs.canonical_smiles,
            cp.mw_freebase as molecular_weight,
            cp.alogp,
            cp.hbd,
            cp.hba,
            cp.psa,
            cp.rtb
        FROM 
            compound_structures cs
        JOIN 
            compound_properties cp ON cs.molregno = cp.molregno
        WHERE 
            cs.canonical_smiles IS NOT NULL
            AND cp.mw_freebase IS NOT NULL
        """
        
        if filter_organics:
            # Add filters for organic molecules
            query += """
            AND cp.mw_freebase BETWEEN 100 AND 800
            AND cp.alogp BETWEEN -3 AND 6
            AND cp.hbd <= 10
            AND cp.hba <= 15
            """
        
        if max_molecules:
            query += f" LIMIT {max_molecules}"
        
        print("Executing query...")
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Retrieved {len(df)} molecules from database")
        
        # Filter by atom count using RDKit
        if min_atoms or max_atoms:
            print("Filtering by atom count...")
            valid_molecules = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
                try:
                    mol = Chem.MolFromSmiles(row['canonical_smiles'])
                    if mol is None:
                        continue
                    
                    num_atoms = mol.GetNumAtoms()
                    
                    if min_atoms and num_atoms < min_atoms:
                        continue
                    if max_atoms and num_atoms > max_atoms:
                        continue
                    
                    valid_molecules.append(idx)
                    
                except Exception:
                    continue
            
            df = df.iloc[valid_molecules].reset_index(drop=True)
            print(f"After filtering: {len(df)} molecules")
        
        return df
    
    def compute_additional_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute additional molecular properties."""
        print("Computing additional molecular properties...")
        
        properties = []
        for smiles in tqdm(df['canonical_smiles'], desc="Computing properties"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    props = {col: None for col in ['num_atoms', 'num_bonds', 'num_rings', 'qed_score']}
                else:
                    props = {
                        'num_atoms': mol.GetNumAtoms(),
                        'num_bonds': mol.GetNumBonds(),
                        'num_rings': mol.GetRingInfo().NumRings(),
                        'qed_score': None  # Would need additional import
                    }
            except Exception:
                props = {col: None for col in ['num_atoms', 'num_bonds', 'num_rings', 'qed_score']}
            
            properties.append(props)
        
        props_df = pd.DataFrame(properties)
        result_df = pd.concat([df, props_df], axis=1)
        
        return result_df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42 # obviously
    ) -> Dict[str, pd.DataFrame]:
        """Split data into train/val/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n_total = len(df_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': df_shuffled[:n_train],
            'val': df_shuffled[n_train:n_train + n_val],
            'test': df_shuffled[n_train + n_val:]
        }
        
        print(f"Data split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files."""
        print("Saving processed data...")
        
        for split_name, split_df in splits.items():
            # Save as TSV
            tsv_path = self.output_dir / f"chembl_{split_name}.tsv"
            split_df.to_csv(tsv_path, sep='\t', index=False)
            
            # Save SMILES only
            smiles_path = self.output_dir / f"chembl_{split_name}_smiles.txt"
            split_df['canonical_smiles'].to_csv(smiles_path, index=False, header=False)
            
            print(f"Saved {split_name}: {len(split_df)} molecules")
        
        # Save summary statistics
        summary = {
            'total_molecules': sum(len(df) for df in splits.values()),
            'splits': {name: len(df) for name, df in splits.items()},
            'properties': {}
        }
        
        # Compute summary statistics for train set
        train_df = splits['train']
        if not train_df.empty:
            numeric_cols = train_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if train_df[col].notna().any():
                    summary['properties'][col] = {
                        'mean': float(train_df[col].mean()),
                        'std': float(train_df[col].std()),
                        'min': float(train_df[col].min()),
                        'max': float(train_df[col].max())
                    }
        
        import json
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Processing complete. Data saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and process ChEMBL data")
    parser.add_argument("--data-dir", default="data/raw/chembl", help="Directory for raw data")
    parser.add_argument("--output-dir", default="data/processed/chembl", help="Directory for processed data")
    parser.add_argument("--max-molecules", type=int, help="Maximum number of molecules to process")
    parser.add_argument("--min-atoms", type=int, default=5, help="Minimum number of atoms")
    parser.add_argument("--max-atoms", type=int, default=50, help="Maximum number of atoms")
    parser.add_argument("--force-redownload", action="store_true", help="Force redownload of files")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    
    args = parser.parse_args()
    
    if not RDKIT_AVAILABLE:
        print("Error: RDKit is required. Install with: pip install rdkit")
        return 1
    
    # Download data
    if not args.skip_download:
        downloader = ChEMBLDownloader(args.data_dir)
        
        if not downloader.download_chembl_data(force_redownload=args.force_redownload):
            print("Failed to download ChEMBL data")
            return 1
        
        if not downloader.extract_files():
            print("Failed to extract files")
            return 1
        
        sqlite_path = downloader.get_sqlite_path()
    else:
        # Look for existing SQLite file
        data_dir = Path(args.data_dir)
        sqlite_files = list(data_dir.glob("*.db")) + list(data_dir.glob("*/*.db"))
        if not sqlite_files:
            print(f"No SQLite database found in {data_dir}")
            return 1
        sqlite_path = sqlite_files[0]
    
    if not sqlite_path or not sqlite_path.exists():
        print("ChEMBL SQLite database not found")
        return 1
    
    print(f"Using SQLite database: {sqlite_path}")
    
    # Process data
    processor = ChEMBLProcessor(sqlite_path, args.output_dir)
    
    # Extract molecules
    df = processor.extract_molecules(
        max_molecules=args.max_molecules,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms
    )
    
    if df.empty:
        print("No molecules extracted")
        return 1
    
    # Compute additional properties
    df = processor.compute_additional_properties(df)
    
    # Split data
    splits = processor.split_data(df)
    
    # Save processed data
    processor.save_processed_data(splits)
    
    print("ChEMBL processing completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
