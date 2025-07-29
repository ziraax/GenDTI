import torch
import torch.nn as nn

class CrossInteractionBlock(nn.Module):
    """
    Cross-interaction block that learns joint representations via elements
    multiplication and residual combination. Inspired by MolTrans-like behavior. 
    """
    def __init__(self, dim):
        super().__init__()
        self.drug_transform = nn.Linear(dim, dim)
        self.prot_transform = nn.Linear(dim, dim)
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, drug, prot):
        d = self.drug_transform(drug)
        p = self.prot_transform(prot)
        interaction = d *p + d + p
        return self.output(interaction)


class DTIModel(nn.Module):
    def __init__(self, drug_dim, protein_dim, fusion, proj_dim=512, hidden_dim=512, dropout=0.3):
        """
        Args:
            drug_dim (int): Dimension of the drug embeddings.
            protein_dim (int): Dimension of the protein embeddings.
            fusion (str): Fusion method for combining drug and protein embeddings. Options: "concat", "sum", "mul".
            proj_dim (int): Dimension of the projection layer.
            hidden_dim (int): Dimension of the hidden layers.
            dropout (float): Dropout rate for regularization.
        """

        super(DTIModel, self).__init__()
        self.fusion = fusion

        # Projection layers to common dimension
        self.proj_drug = nn.Linear(drug_dim, proj_dim)
        self.proj_protein = nn.Linear(protein_dim, proj_dim)
        
        # Initialize projection layers with Xavier initialization
        nn.init.xavier_uniform_(self.proj_drug.weight)
        nn.init.xavier_uniform_(self.proj_protein.weight)
        nn.init.zeros_(self.proj_drug.bias)
        nn.init.zeros_(self.proj_protein.bias)

        # Fusion layer

        if fusion == "concat":
            fusion_dim = proj_dim * 2
        elif fusion == "cross":
            self.cross_block = CrossInteractionBlock(proj_dim)
            fusion_dim = proj_dim *3
        elif fusion in {"sum", "mul"}:
            fusion_dim = proj_dim
        else:
            raise ValueError(f"Invalid fusion method: {fusion}")
        

        # Deep FFN classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize classifier weights properly
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


    def forward(self, drug_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_emb (torch.Tensor): Drug embeddings of shape [B, D_drug].
            protein_emb (torch.Tensor): Protein embeddings of shape [B, D_protein].
        
        Returns:
            torch.Tensor: Predicted binding affinity scores of shape [B, 1].
        """
        # Project embeddings to common dimension
        drug_proj = self.proj_drug(drug_emb)
        protein_proj = self.proj_protein(protein_emb)

        # Fusion
        if self.fusion == "concat":
            fused = torch.cat([drug_proj, protein_proj], dim=-1)
        elif self.fusion == "sum":
            fused = drug_proj + protein_proj
        elif self.fusion == "mul":
            fused = drug_proj * protein_proj
        elif self.fusion == "cross":
            interaction = self.cross_block(drug_proj, protein_proj)
            fused = torch.cat([drug_proj, protein_proj, interaction], dim=-1)
        else:
            raise ValueError(f"Invalid fusion method: {self.fusion}")

        # Classifier
        score = self.classifier(fused)
        return score.squeeze(-1)  # Return raw logits for proper loss calculation