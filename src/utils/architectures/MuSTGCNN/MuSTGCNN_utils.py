import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, average_precision_score  


def best_threshold(y_true, p_pred):
    """Umbral optimo por Youden (J = TPR - FPR) sobre validacion."""
    import numpy as _np
    fpr, tpr, thr = roc_curve(y_true, p_pred)
    return float(thr[_np.argmax(tpr - fpr)])
import optuna    
import torch
import json
from .model import MultimodalGNN_ST as MultimodalGNN
import torch.nn as nn
import time
from optuna.samplers import TPESampler
import torch.nn.functional as F
###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################
import os
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



def saveBestHyperparameters(best_hyperparameters, filename, original_params):
    import copy
    saved_params = copy.deepcopy(original_params)
    for split in best_hyperparameters:
        # Temporal GNN
        temporal_params = {
            "dropout": best_hyperparameters[split]["temporal_dropout"],
            "hid_dim": best_hyperparameters[split]["temporal_hid_dim"],
            "layers": best_hyperparameters[split]["temporal_layers"],
            "in_dim": original_params["temporal_gnn_params"]["in_dim"],
            "out_dim": original_params["temporal_gnn_params"]["out_dim"],
            "embedding_dim": original_params["temporal_gnn_params"]["embedding_dim"]
        }
        # Static GNN
        static_params = {
            "dropout": best_hyperparameters[split]["static_dropout"],
            "hid_dim": best_hyperparameters[split]["static_hid_dim"],
            "layers": best_hyperparameters[split]["static_layers"],
            "in_dim": original_params["static_gnn_params"]["in_dim"],
            "out_dim": original_params["static_gnn_params"]["out_dim"],
            "embedding_dim": original_params["static_gnn_params"]["embedding_dim"]
        }
        # Training params
        train_params = {
            "learning_rate": best_hyperparameters[split]["learning_rate"],
            "decay": best_hyperparameters[split]["decay"],
            "n_epochs": original_params["train_params"]["n_epochs"],
            "early_stopping_patience": original_params["train_params"]["early_stopping_patience"],
        }
        # Save all under split (MuST: sin rnn_params, no hay GRU)
        saved_params[split] = {
            "use_transformer": original_params.get("use_transformer", False),
            "use_static": _get_use_static(original_params),
            "temporal_gnn_params": temporal_params,
            "static_gnn_params": static_params,
            "train_params": train_params
        }
    with open(filename, 'w') as file:
        json.dump(saved_params, file, indent=4)


import random
import copy

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_aware_init(model, X_train_temporal, X_train_static):
    """Initialize attention and fc weights based on data distribution."""
    for name, param in model.named_parameters():
        if 'attention' in name or 'fc' in name:
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data, gain=1.0)
            elif 'bias' in name:
                torch.nn.init.zeros_(param.data)

def loadBestHyperparameters(filename):
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
    Entrenador y optimizador de hiperparámetros para MultimodalGNN_ST sin RNN.
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
        output = dict_output['probability']
        bce_loss = F.binary_cross_entropy_with_logits(dict_output["logits"].squeeze(), y)

        total_loss = bce_loss

        total_loss.backward()
        optimizer.step()

        return total_loss

    def objective(self, trial, S_temporal, S_static,
                 X_train_temporal, X_train_static,
                 X_val_temporal, X_val_static,
                 y_train, y_val, params, seed):
        # Temporal GNN params
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

        # Static GNN params
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

        # Training params
        lr = trial.suggest_float('learning_rate',
                                 low=params['train_params']['learning_rate'][0],
                                 high=params['train_params']['learning_rate'][1],
                                 log=True)
        decay = trial.suggest_float('decay',
                                   low=params['train_params']['decay'][0],
                                   high=params['train_params']['decay'][1],
                                   log=True)

        # Model initialization
        model = MultimodalGNN(
            gnn_params_temporal=temporal_gnn_params,
            gnn_params_static=static_gnn_params,
            S_temporal=S_temporal,
            S_static=S_static,
            debug=False,
            use_static=_get_use_static(params)
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        # Training loop con early stopping
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

    def optimize_hyperparameters(self, S_temporal, S_static,
                                X_train_temporal, X_train_static,
                                X_val_temporal, X_val_static,
                                y_train, y_val, params, seed):
        
        sampler = TPESampler(seed=seed, multivariate=True, group=True)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True
        )
        start = time.time()
        study.optimize(
            lambda trial: self.objective(
                trial, S_temporal, S_static,
                X_train_temporal, X_train_static,
                X_val_temporal, X_val_static,
                y_train, y_val, params, seed
            ),
            n_trials=params['n_trials']
        )
        return study.best_params, time.time() - start







class TestModel:
    def __init__(self, seed, model_class, device,
                 loss_function=nn.BCEWithLogitsLoss(), best_model_path=None):
        self.seed = seed
        self.model_class = model_class
        self.device = device
        self.loss_function = loss_function
        self.best_model_path = best_model_path or f"./model_{seed}.pth"

    def load_hyperparameters(self, hyperparams_path, split):
        with open(hyperparams_path, 'r') as file:
            all_hyperparams = json.load(file)
        return all_hyperparams[split]

    def train_model(self, hyperparams, S_temporal, S_static,
                    X_train_temporal, X_train_static, y_train,
                    X_val_temporal, X_val_static, y_val,
                    n_epochs, early_stopping_patience):
        
        
        hyperparams["temporal_gnn_params"]['seed'] = self.seed
        hyperparams["static_gnn_params"]['seed'] = self.seed
        set_seed(self.seed)
        model = self.model_class(
            gnn_params_temporal=hyperparams["temporal_gnn_params"],
            gnn_params_static=hyperparams["static_gnn_params"],
            S_temporal=S_temporal,
            S_static=S_static,
            use_static=_get_use_static(hyperparams)
        ).to(self.device)
        data_aware_init(model, X_train_temporal, X_train_static)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams["train_params"]["learning_rate"],
            weight_decay=hyperparams["train_params"]["decay"]
        )
        best_roc_auc = -1
        stop = 0
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            dict_output = model(X_train_temporal, X_train_static)
            output = dict_output['probability']
            loss = F.binary_cross_entropy_with_logits(dict_output["logits"].squeeze(), y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                dict_output = model(X_val_temporal, X_val_static)
                val_output = dict_output['probability'].squeeze()
                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.squeeze().cpu().numpy())
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    stop = 0
                    torch.save(model.state_dict(), self.best_model_path)
                else:
                    stop += 1
                    if stop >= early_stopping_patience:
                        break
        model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        # Umbral optimo (Youden) sobre validacion
        model.eval()
        with torch.no_grad():
            p_val = model(X_val_temporal, X_val_static)['probability'].squeeze().cpu().numpy()
        threshold = best_threshold(y_val.cpu().numpy(), p_val)
        return model, None, threshold

    def evaluate(self, model, X_test_temporal, X_test_static, y_test, threshold=0.5):
        model.eval()
        with torch.no_grad():
            start = time.time()
            dict_output = model(X_test_temporal, X_test_static)
            pred_probs = dict_output['probability'].squeeze()
            pred = (pred_probs > threshold).float().view(-1)
            tn, fp, fn, tp = confusion_matrix(y_test.cpu(), pred.cpu()).ravel()
        return {
            "test_acc": (pred == y_test).float().mean().item(),
            "roc_auc": roc_auc_score(y_test.cpu(), pred_probs.cpu()),
            "aucpr": average_precision_score(y_test.cpu(), pred_probs.cpu()),
            "f1_score": f1_score(y_test.cpu(), pred.cpu()),
            "sensitivity": tp/(tp+fn) if tp+fn>0 else 0,
            "specificity": tn/(tn+fp) if tn+fp>0 else 0,
            "inference_time": time.time()-start,
            "threshold": threshold,
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
