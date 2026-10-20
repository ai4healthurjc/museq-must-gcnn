import torch
import torch.nn as nn


class LogisticStaticModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int):

        super().__init__()
        self.total_features = input_dim * seq_len
        self.linear = nn.Linear(self.total_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:

        x_flat = x_seq.view(x_seq.size(0), -1)          
        logits = self.linear(x_flat)                    
        probas = self.sigmoid(logits)                 
        return probas
