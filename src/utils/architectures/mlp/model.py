import torch
import torch.nn as nn

class MLPStaticModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dims: list = [128, 64]):
        """
        input_dim: features per time step
        seq_len: number of time steps
        hidden_dims: hidden dim 
        """
        super().__init__()
        
        total_input = input_dim * seq_len
        layers = []

        prev_dim = total_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid()) 

        self.model = nn.Sequential(*layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch_size, seq_len, input_dim)
        """
        x = x_seq.view(x_seq.size(0), -1)  
        return self.model(x)
