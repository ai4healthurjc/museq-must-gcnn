
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
import torch.nn as nn



class lstmTemporalModel(nn.Module):
    def __init__(self, input_dim, lstm_config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_config["hidden_dim"],
            num_layers=lstm_config["num_layers"],
            batch_first=True,
            dropout=lstm_config["dropout"] if lstm_config["num_layers"] > 1 else 0,
            bidirectional=lstm_config.get("bidirectional", False) 
        )

        self.hidden_dim = lstm_config["hidden_dim"]
        self.bidirectional = lstm_config.get("bidirectional", False)

        output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return torch.sigmoid(self.fc(last_out))