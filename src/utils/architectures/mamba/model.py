import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except Exception as _e:   # pragma: no cover
    Mamba = None
    _MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = _e


class MambaTemporalModel(nn.Module):
    """
    Encoder temporal puro basado en Mamba (SSM), equivalente a GRU/LSTM/Transformer
    como baseline time-variant. Temporal-only (no usa estatico).

    Parameters
    ----------
    input_dim : int
        Numero de features por timestep (F).
    mamba_config : dict
        d_model, d_state, d_conv, expand, dropout (opcional).
    """
    def __init__(self, input_dim, mamba_config):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm no esta instalado. Instala con `pip install mamba-ssm` "
                "(requiere CUDA). Error original: " + str(_MAMBA_IMPORT_ERROR)
            )

        d_model = mamba_config["d_model"]
        d_state = mamba_config.get("d_state", 16)
        d_conv  = mamba_config.get("d_conv", 4)
        expand  = mamba_config.get("expand", 2)
        dropout = mamba_config.get("dropout", 0.1)
        num_layers = mamba_config.get("num_layers", 1)

        # Proyeccion a la dimension del modelo
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Pila de bloques Mamba
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        """
        x = self.input_proj(x)            # (B, T, d_model)
        for layer in self.mamba_layers:
            x = x + layer(x)              # residual, como en stacks de Mamba
            x = self.dropout(x)
        x = self.norm(x)
        last_out = x[:, -1, :]            # (B, d_model) ultimo timestep
        return self.fc(last_out)  # logits (sigmoide se aplica fuera)
