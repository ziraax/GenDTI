import argparse
from pyexpat import model
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
from models.gnn_encoder import GNNEncoder
from data.utils.drug_dataset import DrugDataset
from utils.save_utils import save_embeddings, save_mapping


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DrugDataset(tsv_file=args.input_path, smiles_column=args.smiles_column)
    print(f"Loaded {len(dataset)} unique drug graphs from {args.input_path}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = GNNEncoder(
        input_dim=args.input_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    model.eval()

    all_embeddings = []
    smiles_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            batch = batch.to(device)
            embeddings = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_embeddings.append(embeddings.cpu())
            smiles_list.extend(batch.smiles)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Sanity checks
    assert all_embeddings.shape[0] == len(smiles_list), "Mismatch between embeddings and SMILES count"
    assert not torch.isnan(all_embeddings).any(), "Found NaNs"
    assert not torch.isinf(all_embeddings).any(), "Found Infs"

    save_embeddings(all_embeddings, smiles_list, args.output_path)
    save_mapping(smiles_list, args.output_path)
    
    print(f"Saved {len(smiles_list)} embeddings to {args.output_path}")
    print(f"Final embedding shape: {all_embeddings.shape}")
    print("Drug embedding precomputation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute drug embeddings from SMILES strings.")

    parser.add_argument('--input_path', type=str, required=True, help='Path to the input TSV file with SMILES strings.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the computed embeddings.')
    parser.add_argument('--smiles_column', type=str, default='Ligand_SMILES', help='Column name for SMILES in the TSV file.')

    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension for GNN encoder.')
    parser.add_argument('--edge_dim', type=int, default=5, help='Edge dimension for GNN encoder.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing graphs.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for GNN encoder.')
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension for embeddings.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in GNN encoder.')
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads in GNN encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GNN encoder.')

    args = parser.parse_args()
    main(args)