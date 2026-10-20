import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (⌈d_model/2⌉,)

        # Even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd positions: recortamos si d_model es impar
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            n_odd = pe[:, 1::2].shape[1]  # = d_model // 2
            pe[:, 1::2] = torch.cos(position * div_term[:n_odd])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerTemporalModel(nn.Module):
    def __init__(self, input_dim: int, tf_config: dict):

        super().__init__()

        d_model = tf_config["d_model"]
        nhead = tf_config["nhead"]
        num_layers = tf_config["num_layers"]
        dim_feedforward = tf_config.get("dim_feedforward", 4 * d_model)
        dropout = tf_config.get("dropout", 0.1)
        max_len = tf_config.get("max_len", 5000)

        # Proyección a la dimensión del modelo del Transformer
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        """

        x = self.input_proj(x)  # (batch, seq_len, d_model)

        x = self.pos_encoder(x)

        out = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        last_out = out[:, -1, :]  # (batch, d_model)

        return torch.sigmoid(self.fc(last_out))
