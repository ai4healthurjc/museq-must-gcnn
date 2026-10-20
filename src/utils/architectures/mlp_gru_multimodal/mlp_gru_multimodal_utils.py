import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, average_precision_score
import optuna
import torch.nn as nn
import time
import copy
from typing import Dict, Any
from .model import MultimodalModel
import os
import random
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # si usas múltiples GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)


def load_data(device: torch.device, split: str, numberOfTimeStep:int, SG: str, dataset_name: str, norm: str) -> tuple:
    """
    Load and preprocess static and temporal data for training, validation, and testing.

    Parameters
    ----------
    device : torch.device
        Device to load the data onto (CPU or GPU).
    split : str
        Data split name (e.g., 'train', 'val', 'test').
    SG : bool
        Whether to apply feature-wise temporal averaging and reshape to (n, F×T).

    Returns
    -------
    tuple
        Preprocessed tensors for temporal/static features and labels.
    """
    print('FOLDER INSIDE FUNCTION', split)
    
    # Load static data
    print(f"{DATA_ROOT}/{dataset_name}/{split}/X_train_static_{norm}_target.csv")
    X_train_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_train_static_{norm}_target.csv")
    X_train_static = X_train_static.to_numpy()
    X_train_static = X_train_static.reshape(X_train_static.shape[0], 1, X_train_static.shape[1])
    
    X_val_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_val_static_{norm}_target.csv")
    X_val_static = X_val_static.to_numpy()
    X_val_static = X_val_static.reshape(X_val_static.shape[0], 1, X_val_static.shape[1])

    X_test_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_test_static_{norm}_target.csv")
    X_test_static = X_test_static.to_numpy()
    X_test_static = X_test_static.reshape(X_test_static.shape[0], 1, X_test_static.shape[1])


    # Load temporal data
    X_train_temporal = np.load(f"{DATA_ROOT}/{dataset_name}/{split}/X_train_tensor_{norm}.npy")
    X_val_temporal = np.load(f"{DATA_ROOT}/{dataset_name}/{split}/X_val_tensor_{norm}.npy")
    X_test_temporal = np.load(f"{DATA_ROOT}/{dataset_name}/{split}/X_test_tensor_{norm}.npy")

    # Load labels
    y_train = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/y_train_tensor_{norm}.csv", index_col=0)
    y_val = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/y_val_tensor_{norm}.csv", index_col=0)
    y_test = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/y_test_tensor_{norm}.csv", index_col=0)

    y_train = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/y_train_tensor_{norm}.csv")[['MR_stac']]
    y_train = y_train.iloc[0:y_train.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_train = torch.tensor(y_train['MR_stac'], dtype=torch.float32)


    y_val = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/y_val_tensor_{norm}.csv")[['MR_stac']]
    y_val = y_val.iloc[0:y_val.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_val = torch.tensor(y_val['MR_stac'], dtype=torch.float32)

    y_test = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/y_test_tensor_{norm}.csv")[['MR_stac']]
    y_test = y_test.iloc[0:y_test.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_test = torch.tensor(y_test['MR_stac'], dtype=torch.float32)

    # Replace missing values
    X_train_static[X_train_static == 666] = 0
    X_val_static[X_val_static == 666] = 0
    X_test_static[X_test_static == 666] = 0

    if SG == True:
        # Replace 666 with NaN for temporal
        X_train_temporal[X_train_temporal == 666] = 0
        X_val_temporal[X_val_temporal == 666] = 0
        X_test_temporal[X_test_temporal == 666] = 0

        # Vectorize each of the train/val/test sets
        n, dim1, dim2 = X_train_temporal.shape
        X_train_temporal = X_train_temporal.reshape((n, dim1 * dim2))

        n, dim1, dim2 = X_val_temporal.shape
        X_val_temporal = X_val_temporal.reshape((n, dim1 * dim2))

        n, dim1, dim2 = X_test_temporal.shape
        X_test_temporal = X_test_temporal.reshape((n, dim1 * dim2))

        X_train_temporal = torch.tensor(X_train_temporal, dtype=torch.float32)  # (P, F*T)
        X_val_temporal = torch.tensor(X_val_temporal, dtype=torch.float32)
        X_test_temporal = torch.tensor(X_test_temporal, dtype=torch.float32)

        X_train_temporal = X_train_temporal.unsqueeze(2)
        X_val_temporal = X_val_temporal.unsqueeze(2)
        X_test_temporal = X_test_temporal.unsqueeze(2)

    else:
        # Replace 666 with 0
        X_train_temporal[X_train_temporal == 666] = 0
        X_val_temporal[X_val_temporal == 666] = 0
        X_test_temporal[X_test_temporal == 666] = 0

        # Convert to tensors
        X_train_temporal = torch.tensor(X_train_temporal, dtype=torch.float32)
        X_val_temporal = torch.tensor(X_val_temporal, dtype=torch.float32)
        X_test_temporal = torch.tensor(X_test_temporal, dtype=torch.float32)

        # X_train_temporal = X_train_temporal.unsqueeze(2)
        # X_val_temporal = X_val_temporal.unsqueeze(2)
        # X_test_temporal = X_test_temporal.unsqueeze(2)

    # Convert static
    n, dim1, dim2 = X_train_static.shape
    X_train_static = X_train_static.reshape((n, dim2, dim1))

    n, dim1, dim2 = X_val_static.shape
    X_val_static = X_val_static.reshape((n, dim2, dim1))

    n, dim1, dim2 = X_test_static.shape
    X_test_static = X_test_static.reshape((n, dim2, dim1))

    X_train_static = torch.tensor(X_train_static, dtype=torch.float32)
    X_val_static = torch.tensor(X_val_static, dtype=torch.float32)
    X_test_static = torch.tensor(X_test_static, dtype=torch.float32)

    # Move to device
    if device.type == "cuda":
        return (
            X_train_temporal.to(device), X_val_temporal.to(device), X_test_temporal.to(device),
            X_train_static.to(device), X_val_static.to(device), X_test_static.to(device),
            y_train.to(device), y_val.to(device), y_test.to(device)
        )
    else:
        return (
            X_train_temporal, X_val_temporal, X_test_temporal,
            X_train_static, X_val_static, X_test_static,
            y_train, y_val, y_test
        )
    
    

def saveBestHyperparameters(best_hyperparameters, filename, original_params):
    saved_params = {}

    for split, trial_params in best_hyperparameters.items():
        # Reconstruct MLP
        n_layers = trial_params["mlp_n_layers"]
        hidden_dims = [trial_params[f"mlp_hidden_{i}"] for i in range(n_layers)]
        mlp_params = {
            "static_input_dim": original_params["static_input_dim"],
            "hidden_dims": hidden_dims
        }

        # Reconstruct GRU
        gru_params = {
            "input_dim": original_params["rnn_params"]["input_dim"],
            "hidden_dim": trial_params["hidden_dim"],
            "num_layers": trial_params["num_layers"],
            "dropout": trial_params["dropout"]
        }

        # Reconstruct training parameters
        train_params = {
            "learning_rate": trial_params["learning_rate"],
            "decay": trial_params["decay"],
            "n_epochs": original_params["train_params"]["n_epochs"],
            "early_stopping_patience": original_params["train_params"]["early_stopping_patience"]
        }

        # Store under split
        saved_params[split] = {
            "mlp_params": mlp_params,
            "gru_params": gru_params,
            "train_params": train_params
        }

    # Save to file
    with open(filename, 'w') as f:
        json.dump(saved_params, f, indent=4)


def loadBestHyperparameters(filename):
    with open(filename, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


class Optimizer:
    def __init__(self, device):
        self.device = device
        self.loss_fn = nn.BCELoss()

    def objective(
        self, trial,
        X_tr_s, X_tr_t, y_tr,
        X_val_s, X_val_t, y_val,
        base_params, seed
    ):
        # MLP static branch
        n_layers = trial.suggest_int("mlp_n_layers", base_params['mlp_params']["n_layers"][0], base_params['mlp_params']["n_layers"][1])
        hidden_dims = [
            trial.suggest_int(f"mlp_hidden_{i}", base_params['mlp_params']["hidden_dim"][0], base_params['mlp_params']["hidden_dim"][1])
            for i in range(n_layers)
        ]

        # GRU temporal branch
        gru_config = {
            'hidden_dim': trial.suggest_int('hidden_dim', 
                                                low=base_params['rnn_params']['hidden_dim'][0],
                                                high=base_params['rnn_params']['hidden_dim'][1]),
            'num_layers': trial.suggest_int('num_layers', 
                                                low=base_params['rnn_params']['num_layers'][0],
                                                high=base_params['rnn_params']['num_layers'][1]),
            
            'dropout': trial.suggest_float('dropout', 
                                            low=base_params['rnn_params']['dropout'][0],
                                            high=base_params['rnn_params']['dropout'][1]),
           
            'input_dim': base_params['rnn_params']['input_dim']
        }

        # Optimizer settings
        lr = trial.suggest_float(
            "learning_rate",
            low=base_params['train_params']['learning_rate'][0],
            high=base_params['train_params']['learning_rate'][1],
            log=True
        )
        decay = trial.suggest_float(
            "decay",
            low=base_params['train_params']['decay'][0],
            high=base_params['train_params']['decay'][1],
            log=True
        )

        # Build model
        set_seed(seed)

        model = MultimodalModel(
            static_input_dim=base_params["static_input_dim"],
            static_seq_len=1,
            static_hidden_dims=hidden_dims,
            temporal_input_dim=gru_config["input_dim"],
            gru_config=gru_config,
            fused_dim=base_params["fused_dim"],
            fc_hidden=base_params.get("fc_hidden", 64)
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        best_auc = 0.0
        early_stop_counter = 0

        for epoch in range(base_params['train_params']['n_epochs']):
            model.train()
            optimizer.zero_grad()
            pred, _ = model(X_tr_t, X_tr_s)
            loss = self.loss_fn(pred.squeeze(), y_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t, X_val_s)
                auc = roc_auc_score(y_val.cpu(), val_pred.squeeze().cpu())
                if auc > best_auc:
                    best_auc = auc
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= base_params['train_params']['early_stopping_patience']:
                    break

        return best_auc


    def optimize_hyperparameters(
        self,
        seed:int,
        X_train_temporal: torch.Tensor,
        X_train_static: torch.Tensor,
        X_val_temporal: torch.Tensor,
        X_val_static: torch.Tensor,
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        params: Dict[str, Any],
        n_trials: int
    ):
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        start_time = time.time()

        study.optimize(
            lambda trial: self.objective(
                trial,
                X_train_static, X_train_temporal,
                y_train,
                X_val_static, X_val_temporal,
                y_val,
                params,
                seed
            ),
            n_trials=n_trials
        )

        optimization_time = time.time() - start_time
        return study.best_params, optimization_time


class TestModel:
    def __init__(self, seed, model_class, device, loss_function=nn.BCELoss(), best_model_path=None):
        self.seed = seed
        self.model_class = model_class
        self.device = device
        self.loss_function = loss_function
        self.best_model_path = best_model_path or f"./models/model_{seed}.pth"

    def load_hyperparameters(self, hyperparams_path, split):
        with open(hyperparams_path, 'r') as file:
            all_hyperparams = json.load(file)
        return all_hyperparams[split]

    def train_model(self, flat_params,
                    X_train_temporal, X_train_static, y_train,
                    X_val_temporal, X_val_static, y_val,
                    n_epochs, early_stopping_patience):

        set_seed(self.seed)
        # MLP static encoder
        hidden_dims = flat_params["mlp_params"]["hidden_dims"]

        # GRU temporal encoder
        gru_config = {
            "input_dim": flat_params["gru_params"]["input_dim"],
            "hidden_dim": flat_params["gru_params"]["hidden_dim"],
            "num_layers": flat_params["gru_params"]["num_layers"],
            "dropout": flat_params["gru_params"]["dropout"]
        }

        model = self.model_class(
            static_input_dim=flat_params["mlp_params"]["static_input_dim"],
            static_seq_len=1,
            static_hidden_dims=hidden_dims,
            temporal_input_dim=flat_params["gru_params"]["input_dim"],
            gru_config=gru_config,
            fused_dim=flat_params.get("fused_dim", 64),
            fc_hidden=flat_params.get("fc_hidden", 32)
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=flat_params["train_params"]["learning_rate"],
            weight_decay=flat_params["train_params"]["decay"]
        )

        best_roc_auc = -1
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            output, _ = model(X_train_temporal, X_train_static)
            loss = self.loss_function(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output, _ = model(X_val_temporal, X_val_static)
                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.squeeze().cpu().numpy())

                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    early_stop_counter = 0
                    torch.save(model.state_dict(), self.best_model_path)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        training_time = time.time() - start_time
        model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        return model, training_time


    def evaluate(self, model, X_test_temporal, X_test_static, y_test):
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            pred_probs, weights = model(X_test_temporal, X_test_static)
            end_time = time.time()

            inference_time = end_time - start_time
            pred_probs = pred_probs.squeeze()
            pred = torch.round(pred_probs).view(-1)

        mean_weight = weights.mean().item()
        test_acc = (pred == y_test).float().mean().item()
        roc_auc = roc_auc_score(y_test.cpu().numpy(), pred_probs.cpu().numpy())
        pr_auc = average_precision_score(y_test.cpu().numpy(), pred_probs.cpu().numpy())
        f1 = f1_score(y_test.cpu().numpy(), pred.cpu().numpy())

        tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), pred.cpu().numpy()).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            "test_acc": test_acc,
            "roc_auc": roc_auc,
            "aucpr": pr_auc,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "inference_time": inference_time,
            "mean_weight": mean_weight,
            "pred_probs": pred_probs.cpu().numpy().tolist(),
            "pred_labels": pred.cpu().numpy().tolist()
        }

    def test(self, hyperparams_path, split,
            X_train_temporal, X_train_static, y_train,
            X_val_temporal, X_val_static, y_val,
            X_test_temporal, X_test_static, y_test,
            n_epochs, early_stopping_patience):

        set_seed(self.seed)
        hyperparams = self.load_hyperparameters(hyperparams_path, split)

        print(hyperparams)

        # Detect format and patch missing values
        if "gru_params" in hyperparams and "mlp_params" in hyperparams:
            # Anidado (correcto)
            hyperparams["fused_dim"] = hyperparams.get("fused_dim", 64)
            hyperparams["fc_hidden"] = hyperparams.get("fc_hidden", 32)
        else:
            # Plano (antiguo o incorrecto para este flujo)
            raise ValueError("Expected nested hyperparameter format with 'gru_params' and 'mlp_params' keys.")

        model, training_time = self.train_model(
            hyperparams,
            X_train_temporal, X_train_static, y_train,
            X_val_temporal, X_val_static, y_val,
            n_epochs, early_stopping_patience
        )

        results = self.evaluate(model, X_test_temporal, X_test_static, y_test)
        results["training_time"] = training_time
        return results