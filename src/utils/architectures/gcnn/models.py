import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class StandardGCNNLayer(nn.Module):
    def __init__(self, S: torch.Tensor, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim, self.out_dim = in_dim, out_dim
        S = S.clone()
        S += torch.eye(S.size(0), device=S.device)
        d = S.sum(1)
        eps = 1e-6
        D_inv = torch.diag(1 / torch.sqrt(d + eps))
        S_norm = D_inv @ S @ D_inv
        self.register_buffer('S', S_norm)
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.W, nonlinearity='relu')
        self.b = nn.Parameter(torch.empty(out_dim))
        std = 1 / (in_dim * out_dim)
        nn.init.uniform_(self.b, -std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.S @ x @ self.W + self.b
    
    

class StandardGCNN(nn.Module):
    def __init__(self, layers: int, in_dim: int, hid_dim: int, out_dim: int,
                 dropout: float, seed: int, S: torch.Tensor):
        super().__init__()
        self.layers, self.dropout = layers, dropout
        dims = [in_dim] + ([hid_dim] * (layers - 1)) + [out_dim]
        self.convs = nn.ModuleList([
            StandardGCNNLayer(S, dims[i], dims[i+1], seed)
            for i in range(layers)
        ])
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = self.act(conv(x))
            x = F.dropout(x, self.dropout, training=self.training)
        return self.convs[-1](x)



class TemporalOnlyModel(nn.Module):
    """Temporal-only graph baseline: one GCNN per time step over its own
    adjacency A_t (one scalar output per node), node outputs of all time steps
    concatenated and classified by an MLP head."""
    def __init__(self, gnn_params: Dict, S_temporal: List[torch.Tensor]):
        super().__init__()
        if isinstance(S_temporal, torch.Tensor):
            S_temporal = [S_temporal]
        N = S_temporal[0].shape[0]                 # F
        T = gnn_params['time_steps']               # T
        if len(S_temporal) == 1 and T > 1:         # single graph shared by all steps
            S_temporal = S_temporal * T
        self.gcnns = nn.ModuleList([
            StandardGCNN(
                layers=gnn_params['n_layers'],
                in_dim=gnn_params['in_dim'],
                hid_dim=gnn_params['hid_dim'],
                out_dim=1,
                dropout=gnn_params['dropout'],
                seed=gnn_params['seed'],
                S=S
            ) for S in S_temporal
        ])
        self.classifier = nn.Sequential(
            nn.Linear(T * N, gnn_params['fc_dim']),
            nn.LeakyReLU(),
            nn.Dropout(gnn_params['dropout']),
            nn.Linear(gnn_params['fc_dim'], 1),
        )

    def forward(self, x_temporal: torch.Tensor):
        # x_temporal: [B, T, N]
        B, T, N = x_temporal.shape
        node_feats = torch.stack(
            [gcnn(x_temporal[:, t, :].unsqueeze(-1)).squeeze(-1)
             for t, gcnn in enumerate(self.gcnns)], dim=1)   # [B, T, N]
        flat = node_feats.reshape(B, T * N)
        out = torch.sigmoid(self.classifier(flat))
        return out, node_feats


class StaticOnlyModel(nn.Module):
    def __init__(self,
                 gnn_params: Dict,
                 S_static: torch.Tensor):
        super().__init__()
        N = gnn_params['num_nodes']   
     
        self.gcnn = StandardGCNN(
            layers=gnn_params['n_layers'],
            in_dim=gnn_params['in_dim'], 
            hid_dim=gnn_params['hid_dim'],
            out_dim=1,                    
            dropout=gnn_params['dropout'],
            seed=gnn_params['seed'],
            S=S_static
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(N, gnn_params['fc_dim']),
            nn.LeakyReLU(),
            nn.Dropout(gnn_params['dropout']),
            nn.Linear(gnn_params['fc_dim'], 1),
        )

    def forward(self, x_static: torch.Tensor):
        # Expect x_static: [B, N, F]
        B, N, F = x_static.shape
        # a) GCN per graph → [B, N, 1]
        node_feats = self.gcnn(x_static)
        # b) flatten node dimension → [B, N]
        flat = node_feats.view(B, N)
        # c) classification → [B, 1]
        out = torch.sigmoid(self.classifier(flat))
        return out, node_feats