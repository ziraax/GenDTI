import torch 
from transformers import EsmTokenizer, EsmModel

class ESMEncoder(torch.nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device="cuda"):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, sequences):
        """
        Args:
            sequences: List[str] - amino acid sequences
        Returns: 
            Tensor [B, D] - mean pooled embeddings per protein 
        """
        # Tokenize with padding 
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state # [B, L, D]

        # Remove CLS and EOS tokens (assumes 1 at each end)
        last_hidden = last_hidden[:, 1:-1, :] 
        attention_mask = attention_mask[:, 1:-1]
    
        # Mean pooling 
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # Avoid division by zero
        mean_pooled = summed / counts
        return mean_pooled
    

