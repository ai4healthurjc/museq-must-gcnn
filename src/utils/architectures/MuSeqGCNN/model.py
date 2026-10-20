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


class MultimodalGNN(nn.Module):
    """
    MuSeq-GCNN. At each time step a GCNN is applied to the per-step graph and
    the node embeddings are concatenated, z_t = vec(H_t^T) in R^{F*U}, so each
    variable keeps its own identity. The sequence {z_t} is processed by a GRU
    whose hidden states are averaged over time. The static branch concatenates
    its node embeddings, u = vec(H_s^T) in R^{D*U_s}.
    """
    def __init__(self, gnn_params_temporal: Dict, gnn_params_static: Dict,
                 rnn_params: Dict, S_temporal: List[torch.Tensor],
                 S_static: torch.Tensor, use_transformer: bool = False, debug: bool = False,
                 use_static: bool = True):
        super().__init__()
        self.debug = debug
        self.use_transformer = use_transformer
        self.use_static = use_static

        Ht = gnn_params_temporal['out_dim']
        Hs = gnn_params_static['out_dim']
        Hr = rnn_params['hidden_dim'] * (2 if rnn_params['bidirectional'] else 1)

        self.temporal_gnns = nn.ModuleList([
            StandardGCNN(gnn_params_temporal, S) for S in S_temporal
        ])

        # --- readout dimensions (concatenation) ---
        self.F_nodes = S_temporal[0].shape[0]      # nº variables (nodos temporales)
        self.D_nodes = S_static.shape[0]           # nº nodos estaticos
        temporal_gru_in = Ht * self.F_nodes        # entrada al GRU (por-variable)
        static_flat_dim = Hs * self.D_nodes        # estatico concatenado

        self.rnn = nn.GRU(
            input_size=temporal_gru_in,
            hidden_size=rnn_params['hidden_dim'],
            num_layers=rnn_params['num_layers'],
            bidirectional=rnn_params['bidirectional'],
            dropout=rnn_params['dropout'] if rnn_params['num_layers'] > 1 else 0.0,
            batch_first=True
        )
        self.temporal_out_dim = Hr

        self.static_gnn = StandardGCNN(gnn_params_static, S_static)

        fused_dim = Hr + static_flat_dim
        self.attention = AttentionModule(
            static_dim=static_flat_dim,
            temporal_dim=Hr,
            fused_dim=fused_dim
        )
        self.fc = nn.Linear(Hr + static_flat_dim, 1)
        self.fc_temporal_only = nn.Linear(Hr, 1)

    def _print_debug(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x_temporal_list: List[torch.Tensor], x_static: torch.Tensor):
        temporal_embeddings = []
        for gnn, x_t in zip(self.temporal_gnns, x_temporal_list):
            node_emb = gnn(x_t)                                  # (B, F, Ht)
            g_emb = node_emb.reshape(node_emb.shape[0], -1)      # (B, F*Ht)
            temporal_embeddings.append(g_emb)

        x_seq = torch.stack(temporal_embeddings, dim=1)          # (B, T, F*Ht)
        rnn_out, h_n = self.rnn(x_seq)
        temporal_embedding = rnn_out.mean(dim=1)                 # (B, Hr)

        if self.use_static:
            stat_nodes = self.static_gnn(x_static)               # (B, D, Hs)
            static_embedding = stat_nodes.reshape(stat_nodes.shape[0], -1)  # (B, D*Hs)
            fused, alpha = self.attention(static_embedding, temporal_embedding)
            importance_pre_fc = fused.clone()
            logit = self.fc(fused)
        else:
            static_embedding = None
            importance_pre_fc = temporal_embedding.clone()
            logit = self.fc_temporal_only(temporal_embedding)
            alpha = None

        prob = torch.sigmoid(logit)
        w_fc1 = self.fc.weight.data.clone()
        return {
            'probability': prob,
            'logits': logit,
            'attention_alpha': alpha,
            'temporal_embedding': temporal_embedding,
            'fc1_weights': w_fc1,
            'importance_pre_fc': importance_pre_fc,
            'static_embedding': static_embedding if self.use_static else None
        }
