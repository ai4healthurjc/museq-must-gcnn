import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, average_precision_score, roc_curve  
import optuna    
from optuna.samplers import TPESampler
import torch
import json
from .model import MultimodalGNN
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def best_threshold(y_true, p_pred):
    """Umbral optimo por indice de Youden (J = TPR - FPR) sobre validacion."""
    fpr, tpr, thr = roc_curve(y_true, p_pred)
    return float(thr[np.argmax(tpr - fpr)])

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")

def _get_use_static(params: dict) -> bool:
    """Return the ``use_static`` flag, which must be declared explicitly.

    A silent default would turn a multimodal model into a temporal-only one
    without any warning, so a missing key is treated as a configuration error.
    """
    if "use_static" not in params:
        raise KeyError(
            "'use_static' is not defined for this dataset in the configuration "
            "file. Set it explicitly to true (multimodal) or false "
            "(temporal-only branch)."
        )
    return bool(params["use_static"])


###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################

def saveBestHyperparameters(best_hyperparameters, filename, original_params):
    import copy
    saved_params = copy.deepcopy(original_params)

    for split in best_hyperparameters:
        

        # Temporal GNN
        temporal_params = {
            "dropout": best_hyperparameters[split]["temporal_dropout"],
            "hid_dim": best_hyperparameters[split]["temporal_hid_dim"],
            "layers": best_hyperparameters[split]["temporal_layers"],
            # "seed": best_hyperparameters[split]["seed"],
            "in_dim": original_params["temporal_gnn_params"]["in_dim"],
            "out_dim": original_params["temporal_gnn_params"]["out_dim"],
            "embedding_dim": original_params["temporal_gnn_params"]["embedding_dim"]
        }

        # Static GNN
        static_params = {
            "dropout": best_hyperparameters[split]["static_dropout"],
            "hid_dim": best_hyperparameters[split]["static_hid_dim"],
            "layers": best_hyperparameters[split]["static_layers"],
            # "seed": best_hyperparameters[split]["seed"],
            "in_dim": original_params["static_gnn_params"]["in_dim"],
            "out_dim": original_params["static_gnn_params"]["out_dim"],
            "embedding_dim": original_params["static_gnn_params"]["embedding_dim"]
        }

        # RNN or Transformer
        use_transformer = original_params["use_transformer"]
        if use_transformer == True:
            rnn_params = {
                "d_state": best_hyperparameters[split]["transformer_d_state"],
                "d_conv": best_hyperparameters[split]["transformer_d_conv"],
                "expand": best_hyperparameters[split]["transformer_expand"],
                "in_dim": original_params["rnn_params"]["in_dim"],
                "n_head": best_hyperparameters[split]["n_head"],
                "num_layers": best_hyperparameters[split]["num_layers"],
            }
        else:
            rnn_params = {
                "hidden_dim": best_hyperparameters[split]["rnn_hidden_dim"],
                "num_layers": best_hyperparameters[split]["rnn_num_layers"],
                "bidirectional": best_hyperparameters[split]["rnn_bidirectional"],
                "dropout": best_hyperparameters[split]["rnn_dropout"],
                "in_dim": original_params["rnn_params"]["in_dim"],
                
            }

        # Training params
        train_params = {
            "learning_rate": best_hyperparameters[split]["learning_rate"],
            "decay": best_hyperparameters[split]["decay"],
            "n_epochs": original_params["train_params"]["n_epochs"],
            "early_stopping_patience": original_params["train_params"]["early_stopping_patience"],
            #"lambda_vcl": best_hyperparameters[split]["lambda_vcl"],
        }

        # Save all under split
        saved_params[split] = {
            "use_transformer": use_transformer,
            "use_static": _get_use_static(original_params),
            "temporal_gnn_params": temporal_params,
            "static_gnn_params": static_params,
            "rnn_params": rnn_params,
            "train_params": train_params
        }

    with open(filename, 'w') as file:
        json.dump(saved_params, file, indent=4)
        
def loadBestHyperparameters(filename):
    """
    Load the best hyperparameters from a JSON file.
    
    Args:
        filename (str): The name of the file to load the hyperparameters from.
    
    Returns:
        dict: The dictionary containing the loaded hyperparameters.
    """
    with open(filename, 'r') as file:
        hyperparameters = json.load(file)
        
    return hyperparameters

#########################
# Function to load data #
#########################

def load_data(device: torch.device, split: str, numberOfTimeStep:int, SG: str, dataset_name: str, norm: str, norm_static: str = None) -> tuple:
    if norm_static is None:
        norm_static = norm
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
    
    # Load static data (OPCIONAL: temporal-only si no existe)
    _static_path = f"{DATA_ROOT}/{dataset_name}/{split}/X_train_static_{norm_static}.csv"
    has_static = os.path.exists(_static_path)
    if has_static:
        print(_static_path)
        X_train_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_train_static_{norm_static}.csv").to_numpy()
        X_train_static = X_train_static.reshape(X_train_static.shape[0], 1, X_train_static.shape[1])
        X_val_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_val_static_{norm_static}.csv").to_numpy()
        X_val_static = X_val_static.reshape(X_val_static.shape[0], 1, X_val_static.shape[1])
        X_test_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_test_static_{norm_static}.csv").to_numpy()
        X_test_static = X_test_static.reshape(X_test_static.shape[0], 1, X_test_static.shape[1])
    else:
        print(f"[load_data] sin estatico ({_static_path} no existe) -> temporal-only")
        X_train_static = X_val_static = X_test_static = None


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

    # Replace missing values (solo si hay estatico)
    if has_static:
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

    # Convert static (solo si hay estatico)
    if has_static:
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
            (X_train_static.to(device) if has_static else None),
            (X_val_static.to(device) if has_static else None),
            (X_test_static.to(device) if has_static else None),
            y_train.to(device), y_val.to(device), y_test.to(device)
        )
    else:
        return (
            X_train_temporal, X_val_temporal, X_test_temporal,
            X_train_static, X_val_static, X_test_static,
            y_train, y_val, y_test
        )
    

    

class Optimizer:
    """
    A class for training models and performing hyperparameter optimization.
    """
    def __init__(self, device, loss_function=nn.BCEWithLogitsLoss(), storage=None, study_name=None):
        self.device = device
        self.loss_function = loss_function
        self.best_model_path = "best_multimodal_model.pth"
        self.storage = storage
        self.study_name = study_name

    def train(self, model, X_temporal, X_static, y, optimizer):
        model.train()
        optimizer.zero_grad()

        dict_output = model(X_temporal, X_static)
        logits = dict_output['logits'].squeeze()
        bce_loss = F.binary_cross_entropy_with_logits(logits, y)

        total_loss = bce_loss

        total_loss.backward()
        optimizer.step()

        return total_loss

    def objective(self, trial, S_temporal_list, S_static, 
                 X_train_temporal, X_train_static,
                 X_val_temporal, X_val_static,
                 y_train, y_val, params, seed):
        # Temporal GNN parameters
        temporal_gnn_params = {
            'dropout': trial.suggest_float('temporal_dropout', 
                                          low=params['temporal_gnn_params']['dropout'][0],
                                          high=params['temporal_gnn_params']['dropout'][1]),
            'hid_dim': trial.suggest_int('temporal_hid_dim', 
                                       low=params['temporal_gnn_params']['hid_dim'][0],
                                       high=params['temporal_gnn_params']['hid_dim'][1]),
            'layers': trial.suggest_int('temporal_layers', 
                                       low=params['temporal_gnn_params']['layers'][0],
                                       high=params['temporal_gnn_params']['layers'][1]),
            'in_dim': params['temporal_gnn_params']['in_dim'],
            'out_dim': params['temporal_gnn_params']['out_dim'],
            'seed': seed,
            'embedding_dim': params['temporal_gnn_params']['embedding_dim']
        }

        # Static GNN parameters
        static_gnn_params = {
            'dropout': trial.suggest_float('static_dropout', 
                                         low=params['static_gnn_params']['dropout'][0],
                                         high=params['static_gnn_params']['dropout'][1]),
            'hid_dim': trial.suggest_int('static_hid_dim', 
                                       low=params['static_gnn_params']['hid_dim'][0],
                                       high=params['static_gnn_params']['hid_dim'][1]),
            'layers': trial.suggest_int('static_layers', 
                                       low=params['static_gnn_params']['layers'][0],
                                       high=params['static_gnn_params']['layers'][1]),
            'in_dim': params['static_gnn_params']['in_dim'],
            'out_dim': params['static_gnn_params']['out_dim'],
            'seed': seed,
            'embedding_dim': params['static_gnn_params']['embedding_dim']
        }

        use_transformer = params['use_transformer']
        if use_transformer == False:
            rnn_params = {
                'hidden_dim': trial.suggest_int('rnn_hidden_dim', 
                                                low=params['rnn_params']['hidden_dim'][0],
                                                high=params['rnn_params']['hidden_dim'][1]),
                'num_layers': trial.suggest_int('rnn_num_layers', 
                                                low=params['rnn_params']['num_layers'][0],
                                                high=params['rnn_params']['num_layers'][1]),
                'bidirectional': trial.suggest_categorical('rnn_bidirectional', 
                                                        params['rnn_params']['bidirectional']),
                'dropout': trial.suggest_float('rnn_dropout', 
                                            low=params['rnn_params']['dropout'][0],
                                            high=params['rnn_params']['dropout'][1]),
                'in_dim': params['rnn_params']['in_dim'],
                'embedding_dim': params['temporal_gnn_params']['embedding_dim']
            }
        else:
            rnn_params = {
                'in_dim': params['rnn_params']['in_dim'], 
                'num_layers': trial.suggest_int('num_layers', 
                                                low=params['rnn_params']['num_layers'][0],
                                                high=params['rnn_params']['num_layers'][1]),
                'n_head': trial.suggest_categorical('n_head', params['rnn_params']['n_head']),
                'd_state': trial.suggest_categorical('transformer_d_state', params['rnn_params']['d_state']),
                'd_conv': trial.suggest_categorical('transformer_d_conv', params['rnn_params']['d_conv']),
                'expand': trial.suggest_categorical('transformer_expand', params['rnn_params']['expand'])
            }

        lr = trial.suggest_float('learning_rate', 
                               low=params['train_params']['learning_rate'][0],
                               high=params['train_params']['learning_rate'][1], 
                               log=True)
        decay = trial.suggest_float('decay', 
                                  low=params['train_params']['decay'][0],
                                  high=params['train_params']['decay'][1], 
                                  log=True)

        model = MultimodalGNN(
            gnn_params_temporal=temporal_gnn_params,
            gnn_params_static=static_gnn_params,
            rnn_params=rnn_params,
            S_temporal=S_temporal_list,
            S_static=S_static,
            use_transformer=use_transformer,
            use_static=_get_use_static(params)
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        best_roc_auc = -1
        early_stop_counter = 0

        for epoch in range(params['train_params']['n_epochs']):
            self.train(model, X_train_temporal, X_train_static, y_train, optimizer)

            model.eval()
            with torch.no_grad():
                dict_output = model(X_val_temporal, X_val_static)
                val_output = dict_output['probability'].squeeze()
                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.cpu().numpy())

                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    early_stop_counter = 0
                    torch.save(model.state_dict(), self.best_model_path)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= params['train_params']['early_stopping_patience']:
                    break

        return best_roc_auc

    def optimize_hyperparameters(self, S_temporal_list, S_static,
                                X_train_temporal, X_train_static,
                                X_val_temporal, X_val_static,
                                y_train, y_val, params, seed, n_trials=100):
        import time
        sampler = TPESampler(seed=seed, multivariate=True, group=True)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True
        )

        start_time = time.time()

        study.optimize(
            lambda trial: self.objective(
                trial, S_temporal_list, S_static,
                X_train_temporal, X_train_static,
                X_val_temporal, X_val_static,
                y_train, y_val, params, seed
            ),
            n_trials=params['n_trials']
        )

        optimization_time = time.time() - start_time
        return study.best_params, optimization_time



import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import gc

import random

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_aware_init(model: torch.nn.Module, X_train_temporal, X_train_static):
    """
    Initialize GRU and Linear weights based on data distribution.
    Scales weights by the std of the input data to avoid vanishing/exploding gradients.
    """
    # Calcular std de los datos temporales y estáticos
    if isinstance(X_train_temporal, list):
        # Lista de tensores [B, F, 1]
        temporal_tensor = torch.stack([x.squeeze(-1) for x in X_train_temporal], dim=1)  # [B, T, F]
    else:
        temporal_tensor = X_train_temporal
    
    temporal_std = temporal_tensor.float().std().item()
    temporal_std = max(temporal_std, 1e-6)  # evitar división por cero
    
    if X_train_static is not None:
        static_std = X_train_static.float().std().item()
        static_std = max(static_std, 1e-6)
    
    for name, param in model.named_parameters():
        if 'rnn' in name or 'gru' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data, gain=temporal_std)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        elif 'attention' in name or 'fc' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


class TestModel:
    def __init__(self, seed, model_class, device, loss_function=torch.nn.BCEWithLogitsLoss(), best_model_path=None):
        self.seed = seed
        self.model_class = model_class
        self.device = device
        self.loss_function = loss_function
        self.best_model_path = best_model_path or f"./models/rnn_gnn_1/model_{seed}.pth"

    def load_hyperparameters(self, hyperparams_path, split):
        with open(hyperparams_path, 'r') as file:
            all_hyperparams = json.load(file)
        return all_hyperparams[split]

    def train_model(self, hyperparams, S_temporal_list, S_static,
                    X_train_temporal, X_train_static, y_train,
                    X_val_temporal, X_val_static, y_val,
                    n_epochs, early_stopping_patience):
        import time
        hyperparams["temporal_gnn_params"]['seed'] = self.seed
        hyperparams["static_gnn_params"]['seed'] = self.seed
        # Fix seed before model creation for reproducibility
        set_seed(self.seed)
        model = self.model_class(
            gnn_params_temporal=hyperparams["temporal_gnn_params"],
            gnn_params_static=hyperparams["static_gnn_params"],
            rnn_params=hyperparams["rnn_params"],
            S_temporal=S_temporal_list,
            S_static=S_static,
            use_transformer=hyperparams.get("use_transformer", False),
            use_static=_get_use_static(hyperparams)
        ).to(self.device)
        # Data-aware initialization
        data_aware_init(model, X_train_temporal, X_train_static)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyperparams["train_params"]["learning_rate"],
                                     weight_decay=hyperparams["train_params"]["decay"])

        best_roc_auc = -1
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            dict_output = model(X_train_temporal, X_train_static)
            logits = dict_output['logits'].squeeze()
            bce_loss = self.loss_function(logits, y_train)

            total_loss = bce_loss
            total_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                dict_output = model(X_val_temporal, X_val_static)
                val_output = dict_output['probability'].squeeze()

                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.cpu().numpy())

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

        # Umbral optimo sobre VALIDACION (no sobre test) con el mejor modelo cargado
        model.eval()
        with torch.no_grad():
            p_val = model(X_val_temporal, X_val_static)['probability'].squeeze().cpu().numpy()
        threshold = best_threshold(y_val.cpu().numpy(), p_val)

        return model, training_time, threshold

    def evaluate(self, model, X_test_temporal, X_test_static, y_test, threshold=0.5):
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            dict_output = model(X_test_temporal, X_test_static)
            pred_probs = dict_output['probability'].squeeze()
            end_time = time.time()

        pred = (pred_probs > threshold).float().view(-1)
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
            "threshold": threshold,
            "inference_time": end_time-start_time,
            "pred_probs": pred_probs.cpu().numpy().tolist(),
            "pred_labels": pred.cpu().numpy().tolist(),
        }, dict_output

    def test(self, hyperparams_path, split, S_temporal_list, S_static,
             X_train_temporal, X_train_static, y_train,
             X_val_temporal, X_val_static, y_val,
             X_test_temporal, X_test_static, y_test,
             n_epochs, early_stopping_patience):
        hyperparams = self.load_hyperparameters(hyperparams_path, split)

        model, training_time, threshold = self.train_model(
            hyperparams, S_temporal_list, S_static,
            X_train_temporal, X_train_static, y_train,
            X_val_temporal, X_val_static, y_val,
            n_epochs, early_stopping_patience
        )

        results, dict_output = self.evaluate(model, X_test_temporal, X_test_static, y_test, threshold=threshold)
        results["training_time"] = training_time
        return results, dict_output

