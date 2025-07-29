# scripts/split_dataset.py

import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_path, train_path, val_path, test_path, val_ratio=0.1, test_ratio=0.1, random_state=42):
    df = pd.read_csv(input_path, sep="\t")

    # First split out test set
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state, shuffle=True)
    # Then split train_val into train and val
    val_size_adjusted = val_ratio / (1 - test_ratio)  # adjust val ratio relative to remaining after test split
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=random_state, shuffle=True)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    print(f"Split dataset into:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"Train saved to {train_path}")
    print(f"Validation saved to {val_path}")
    print(f"Test saved to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation and test sets")
    parser.add_argument("--input_path", type=str, required=True, help="Path to full dataset TSV")
    parser.add_argument("--train_path", type=str, required=True, help="Path to save train TSV")
    parser.add_argument("--val_path", type=str, required=True, help="Path to save validation TSV")
    parser.add_argument("--test_path", type=str, required=True, help="Path to save test TSV")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")

    args = parser.parse_args()
    split_dataset(args.input_path, args.train_path, args.val_path, args.test_path, args.val_ratio, args.test_ratio)
