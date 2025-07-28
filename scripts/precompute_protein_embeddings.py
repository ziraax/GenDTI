import os
import argparse
import pandas as pd 
import torch 
import json
from tqdm import tqdm
from models.esm_encoder import ESMEncoder

def load_dataset(path, sequence_column="BindingDB Target Chain Sequence"):
    """Load unique protein sequences from the dataset. """
    df = pd.read_csv(path, sep="\t")
    sequences = df[sequence_column].dropna().unique().tolist()
    return sequences

def save_mapping(sequences_ids, output_path):
    seq_to_idx = {seq: i for i, seq in enumerate(sequences_ids)}
    map_path = os.path.splitext(output_path)[0] + "_seq_to_idx.json"
    with open(map_path, 'w') as f:
        json.dump(seq_to_idx, f)
    print(f"Saved sequence to index mapping to {map_path}")

def chunked(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def save_embeddings(embeddings, sequence_ids, output_path):
    """Save the dictionary of embeddings to a .pt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"ids": sequence_ids, "embeddings": embeddings}, output_path)
    print(f"Saved {len(sequence_ids)} embeddings to {output_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMEncoder(model_name=args.model_name, device=device)

    sequences = load_dataset(args.input_path, sequence_column=args.sequence_column)
    print(f"Loaded {len(sequences)} unique sequences from {args.input_path}")

    embeddings = []
    sequence_ids = []

    for batch in tqdm(chunked(sequences, args.batch_size), desc="Encoding proteins"):
        reps = model(batch)  # [B, D]
        embeddings.append(reps.cpu())
        sequence_ids.extend(batch)
        torch.cuda.empty_cache()  # Free up GPU memory after each batch

    # Concatenate embeddings for all sequences
    all_embeddings = torch.cat(embeddings, dim=0)  # [N, D]

    # Validations
    embedding_dim = model.model.config.hidden_size
    assert all_embeddings.shape[1] == embedding_dim, f"Embedding dimension mismatch, got {all_embeddings.shape[1]}"
    assert not torch.isnan(all_embeddings).any(), "Found NaNs in embeddings"
    assert not torch.isinf(all_embeddings).any(), "Found Infs in embeddings"
    assert all_embeddings.shape[0] == len(sequence_ids), "Mismatch in number of embeddings and sequences"

    # Save embeddings
    save_embeddings(all_embeddings, sequence_ids, args.output_path)
    save_mapping(sequence_ids, args.output_path)

    print(f"Total embeddings shape: {all_embeddings.shape}")
    print("Protein embeddings precomputation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute protein embeddings using ESM model.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the precomputed embeddings.")
    parser.add_argument("--sequence_column", type=str, default="BindingDB Target Chain Sequence", help="Column name for protein sequences in the dataset.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="HuggingFace model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding sequences.")

    args = parser.parse_args()
    main(args)


