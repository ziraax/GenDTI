import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, input_dim=7, edge_dim=3, hidden_dim=128, output_dim=256, num_layers=3, dropout=0.1):
        """
        NNConv-based GNN encoder that uses both atom and bond features.

        Args:
            input_dim (int): Dimension of atom (node) features.
            edge_dim (int): Dimension of bond (edge) features.
            hidden_dim (int): Hidden layer size.
            output_dim (int): Final graph embedding size.
            num_layers (int): Number of NNConv layers.
            dropout (float): Dropout probability.
        """
        super(GNNEncoder, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.edge_networks = nn.ModuleList()

        # First layer
        self.edge_networks.append(nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
        ))
        self.convs.append(NNConv(input_dim, hidden_dim, self.edge_networks[-1]))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.edge_networks.append(nn.Sequential(
                nn.Linear(edge_dim, hidden_dim * hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
            ))
            self.convs.append(NNConv(hidden_dim, hidden_dim, self.edge_networks[-1]))

        # Final layer
        self.edge_networks.append(nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * output_dim)
        ))
        self.convs.append(NNConv(hidden_dim, output_dim, self.edge_networks[-1]))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        graph_emb = global_mean_pool(x, batch)
        return graph_emb
