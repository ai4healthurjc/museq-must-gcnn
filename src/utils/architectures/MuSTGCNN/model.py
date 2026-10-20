import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Dict


class StandardGCNNLayer(nn.Module):
    """Capa GCN estandar: S_norm @ x @ W + b."""
    def __init__(self, S: torch.Tensor, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.S = S.clone()
        self.S += torch.eye(self.S.shape[0], device=self.S.device)
        self.d = self.S.sum(1)
        eps = 1e-6
        self.D_inv = torch.diag(1 / torch.sqrt(self.d + eps))
        self.S = self.D_inv @ self.S @ self.D_inv
        self.S = nn.Parameter(self.S, requires_grad=False)
        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.W.data, nonlinearity='relu')
        self.b = nn.Parameter(torch.empty(self.out_dim))
        std = 1 / (self.in_dim * self.out_dim)
        nn.init.uniform_(self.b.data, -std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.S @ x @ self.W + self.b[None, :]


class StandardGCNN(nn.Module):
    def __init__(self, gnn_params: Dict, S: torch.Tensor):
        super().__init__()
        self.convs = nn.ModuleList()
        self.nonlin = nn.ReLU()
        self.n_layers = gnn_params['layers']
        self.dropout = gnn_params['dropout']
        in_dim = gnn_params['in_dim']
        hid_dim = gnn_params['hid_dim']
        out_dim = gnn_params['out_dim']
        seed = gnn_params['seed']
        if self.n_layers > 1:
            self.convs.append(StandardGCNNLayer(S, in_dim, hid_dim, seed))
            for _ in range(self.n_layers - 2):
                in_dim = hid_dim
                self.convs.append(StandardGCNNLayer(S, in_dim, hid_dim, seed))
            in_dim = hid_dim
            self.convs.append(StandardGCNNLayer(S, in_dim, out_dim, seed))
        else:
            self.convs.append(StandardGCNNLayer(S, in_dim, out_dim, seed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


class AttentionModule(nn.Module):
    """Gated-attention fusion: linear projections to a common space of size M
    and a patient-specific scalar gate alpha that weights the static branch."""
    def __init__(self, static_dim, temporal_dim, fused_dim):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, fused_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fused_dim)
        self.gate = nn.Sequential(nn.Linear(fused_dim * 2, 1), nn.Sigmoid())

    def forward(self, static, temporal):
        static = self.static_proj(static)
        temporal = self.temporal_proj(temporal)
        concat = torch.cat([static, temporal], dim=-1)
        alpha = self.gate(concat)
        fused = alpha * static + (1 - alpha) * temporal
        return fused, alpha


class MultimodalGNN_ST(nn.Module):
    """
    MuST-GCNN (simultaneous architecture).

    A single GCNN runs over the unified spatio-temporal graph (F*T nodes, ordered
    as index = t*F + f). Its last layer outputs ONE scalar per node, so the node
    embeddings are concatenated directly into z^ST in R^{F*T}: each entry is one
    feature-time pair. The static GCNN likewise outputs one scalar per node,
    giving u in R^{D}. Both are fused with gated attention (M = F*T + D).
    """
    def __init__(self,
                 gnn_params_temporal: Dict,
                 gnn_params_static: Dict,
                 S_temporal: torch.Tensor,
                 S_static: torch.Tensor,
                 use_transformer: bool = False,
                 debug: bool = False,
                 use_static: bool = True):
        super().__init__()
        self.debug = debug
        self.use_transformer = use_transformer
        self.use_static = use_static

        # one scalar output per node (U = 1)
        gnn_params_temporal = {**gnn_params_temporal, 'out_dim': 1}
        gnn_params_static = {**gnn_params_static, 'out_dim': 1}

        self.temporal_gnn = StandardGCNN(gnn_params_temporal, S_temporal)
        self.static_gnn = StandardGCNN(gnn_params_static, S_static)

        temporal_dim = S_temporal.shape[0]   # F*T
        static_dim = S_static.shape[0]       # D
        fused_dim = temporal_dim + static_dim

        self.attention = AttentionModule(static_dim=static_dim,
                                         temporal_dim=temporal_dim,
                                         fused_dim=fused_dim)
        self.fc = nn.Linear(fused_dim, 1)
        self.fc_temporal_only = nn.Linear(temporal_dim, 1)

    def _print_debug(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor):
        temporal_emb = self.temporal_gnn(x_temporal).squeeze(-1)      # (B, F*T)

        if self.use_static:
            static_embedding = self.static_gnn(x_static).squeeze(-1)  # (B, D)
            fused, alpha = self.attention(static_embedding, temporal_emb)
            importance_pre_fc = fused.clone()
            logit = self.fc(fused)
        else:
            static_embedding = None
            alpha = None
            importance_pre_fc = temporal_emb.clone()
            logit = self.fc_temporal_only(temporal_emb)

        prob = torch.sigmoid(logit)
        w_fc1 = self.fc.weight.data.clone()
        return {
            'probability': prob,
            'logits': logit,
            'attention_alpha': alpha,
            'static_embedding': static_embedding,
            'temporal_embedding': temporal_emb,
            'fc1_weights': w_fc1,
            'importance_pre_fc': importance_pre_fc
        }
