import torch
import torch.nn as nn


class GRUTemporalModel(nn.Module):
    def __init__(self, input_dim: int, gru_config: dict):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_config["hidden_dim"],
            num_layers=gru_config["num_layers"],
            batch_first=True,
            dropout=gru_config["dropout"] if gru_config["num_layers"] > 1 else 0,
            bidirectional=gru_config.get("bidirectional", False) 
        )
        self.output = nn.Linear(gru_config["hidden_dim"], 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x_seq)  # (batch, seq_len, hidden_dim)
        x_last = output[:, -1, :]    
        out = torch.sigmoid(self.output(x_last))  # (batch, 1)
        return out


class MLPStaticModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dims: list = [128, 64]):
        super().__init__()
        total_input = input_dim * seq_len
        layers = []
        prev_dim = total_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.model = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)  
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = x_seq.view(x_seq.size(0), -1)  
        features = self.model(x)  # (batch, last_hidden_dim)
        return features


class AttentionModule(nn.Module):
    def __init__(self, static_dim, temporal_dim, fused_dim):
        super().__init__()
        self.static_proj   = nn.Linear(static_dim, fused_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, static_emb, temporal_emb):
        s = self.static_proj(static_emb)     
        t = self.temporal_proj(temporal_emb) 
        concat = torch.cat([s, t], dim=-1)   
        alpha  = self.gate(concat)           
        fused  = alpha * s + (1 - alpha) * t  
        return fused, alpha


class MultimodalModel(nn.Module):
    def __init__(
        self,
        static_input_dim: int,
        static_seq_len: int,
        static_hidden_dims: list,
        temporal_input_dim: int,
        gru_config: dict,
        fused_dim: int,
        fc_hidden: int = 64
    ):
        super().__init__()
        self.static_encoder = MLPStaticModel(
            input_dim=static_input_dim,
            seq_len=static_seq_len,
            hidden_dims=static_hidden_dims
        )
        static_emb_dim = static_hidden_dims[-1]

        self.temporal_encoder = GRUTemporalModel(
            input_dim=temporal_input_dim,
            gru_config=gru_config
        )
        temporal_emb_dim = 1  

        self.attn = AttentionModule(
            static_dim=static_emb_dim,
            temporal_dim=temporal_emb_dim,
            fused_dim=fused_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor):
        static_emb = self.static_encoder(x_static)  

        temporal_emb = self.temporal_encoder(x_seq) 

        fused, alpha = self.attn(static_emb, temporal_emb)

        out = self.classifier(fused)
        return out, alpha

