import models
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, average_precision_score
from typing import Dict, Any, Tuple, List
import time
import sys


import os as _os
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../../.."))


def load_temporal_graphs(dataset_name: str, way_to_build_graph: str, folder: str,
                         norm: str, numberOfTimeStep: int, device,
                         threshold: str = "0.975") -> List[torch.Tensor]:
    """Per-time-step adjacency matrices A_t (exponential kernel + pruning),
    produced by graph_estimation/temporal_data/step2_graphRepresentation."""
    graphs = []
    for t in range(numberOfTimeStep):
        df = pd.read_csv(
            f"{_PROJECT_ROOT}/graph_estimation/temporal_data/step2_graphRepresentation/"
            f"{dataset_name}/{way_to_build_graph}/{folder}/adj_X_train_TS_{t}_{norm}_th_{threshold}.csv"
        )
        graphs.append(torch.tensor(df.fillna(0).values, dtype=torch.float32, device=device))
    return graphs
import os
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")
###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################

def saveBestHyperparameters(best_hyperparameters, filename):
    """
    Save the best hyperparameters to a JSON file.
    
    Args:
        best_hyperparameters (dict): The dictionary containing the best hyperparameters.
        filename (str): The name of the file to save the hyperparameters.
    """
    with open(filename, 'w') as file:
        json.dump(best_hyperparameters, file, indent=4)
        
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
    
    # Load static data (OPCIONAL: temporal-only si no existe)
    _static_path = f"{DATA_ROOT}/{dataset_name}/{split}/X_train_static_{norm}.csv"
    has_static = os.path.exists(_static_path)
    if has_static:
        print(_static_path)
        X_train_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_train_static_{norm}.csv").to_numpy()
        X_train_static = X_train_static.reshape(X_train_static.shape[0], 1, X_train_static.shape[1])
        X_val_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_val_static_{norm}.csv").to_numpy()
        X_val_static = X_val_static.reshape(X_val_static.shape[0], 1, X_val_static.shape[1])
        X_test_static = pd.read_csv(f"{DATA_ROOT}/{dataset_name}/{split}/X_test_static_{norm}.csv").to_numpy()
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

    # Replace missing values
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



    
############################
# Function to train models #
############################

import optuna
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score

class Optimizer:
    def __init__(self):
        self.optimizer = None

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def train(self, model, X_temporal, _, y, loss_function):
        model.train()
        self.zero_grad()
        output, _ = model(X_temporal)
        output = output.squeeze()
        loss = loss_function(output, y)
        loss.backward()
        self.step()
        return loss

    def optuna_objective(
        self,
        folder: str,
        trial,
        search_space: dict,
        S_temporal: torch.Tensor,
        S_static: torch.Tensor,
        X_train_temporal: torch.Tensor,
        X_train_static: torch.Tensor,
        X_val_temporal: torch.Tensor,
        X_val_static: torch.Tensor,
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        params: Dict[str, Any],
        device: torch.device,
        seed:int,
    ) -> float:
        sampled_params = {}
        for param_name, spec in search_space.items():
            if spec["type"] == "float":
                sampled_params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "int":
                sampled_params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                sampled_params[param_name] = trial.suggest_categorical(param_name, spec["choices"])

        sampled_params['num_nodes'] = S_temporal[0].size(0)
        sampled_params['in_dim'] = 1
        sampled_params['time_steps'] = params["numberOfTimeStep"]
        sampled_params['seed'] = seed
        print(sampled_params)
        model = models.TemporalOnlyModel(gnn_params=sampled_params,
                                         S_temporal=S_temporal
        ).to(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=sampled_params['lr'], weight_decay=sampled_params['decay'])
        loss_function = nn.BCELoss()

        best_val_roc = 0
        early_stop_counter = 0

        for epoch in range(params["n_epochs"]):
            self.train(model, X_train_temporal, None, y_train, loss_function)

            model.eval()
            with torch.no_grad():
                output_val, _ = model(X_val_temporal)
                output_val = output_val.squeeze()
                roc_auc = roc_auc_score(y_val.cpu().numpy(), output_val.cpu().numpy())

                if roc_auc > best_val_roc:
                    best_val_roc = roc_auc
                    early_stop_counter = 0
                    torch.save(model, f"saved_models/best_model_{folder}.pt")
                else:
                    early_stop_counter += 1

                if early_stop_counter >= params["early_stopping_patience"]:
                    break

        return best_val_roc
    
    def run_optuna(
        self,
        folder: str,
        seed:int,
        S_temporal: torch.Tensor,
        S_static: torch.Tensor,
        X_train_temporal: torch.Tensor,
        X_train_static: torch.Tensor,
        X_val_temporal: torch.Tensor,
        X_val_static: torch.Tensor,
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        params: Dict[str, Any],
        device: torch.device,
        search_space: Dict[str, Any],
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: self.optuna_objective(
            folder,
            trial,
            search_space,
            S_temporal,
            S_static,
            X_train_temporal,
            X_train_static,
            X_val_temporal,
            X_val_static,
            y_train,
            y_val,
            params,
            device,
            seed
        ), n_trials=n_trials)

        return study.best_params


class Tester:
    """
    Class to train and evaluate a temporal-only GNN using best hyperparameters.
    """
    def __init__(self, best_params_file: str):
        # Load best hyperparameters per split from JSON
        import json
        with open(best_params_file, 'r') as f:
            self.best_params_by_folder: Dict[str, Dict[str, Any]] = json.load(f)

    def train_and_save(
        self,
        params: Dict[str, Any],
        device: torch.device,
        train_epochs: int = 1000
    ) -> Dict[str, float]:
        """
        Train the temporal model for each folder, save its state_dict,
        and return a dict of training times per folder.
        """
        training_times: Dict[str, float] = {}

        for folder, hparams in self.best_params_by_folder.items():
            torch.cuda.empty_cache()

            # Load training temporal data
            X_train_temporal, _, _, _, _, _, y_train, _, _ = \
                load_data(device, folder, params["numberOfTimeStep"], params["SG"], params["dataset_name"], params["norm"])

            # Load per-time-step adjacency matrices (kernel + pruning)
            S_temporal = load_temporal_graphs(params['dataset_name'], params['way_to_build_graph'],
                                              folder, params['norm'], params['numberOfTimeStep'], device)

            # Add required init params
            hparams['num_nodes'] = S_temporal[0].size(0)
            hparams['in_dim'] = 1
            hparams['time_steps'] = params["numberOfTimeStep"]
            hparams['seed'] = params['seeds'][int(folder[1:])-1]

            # Instantiate model
            model = models.TemporalOnlyModel(
                gnn_params=hparams,
                S_temporal=S_temporal
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hparams.get('lr', 1e-3),
                weight_decay=hparams.get('decay', 0)
            )
            loss_fn = nn.BCELoss()

            # Measure training time
            start = time.time()
            model.train()
            for epoch in range(train_epochs):
                optimizer.zero_grad()
                output, _ = model(X_train_temporal)
                loss = loss_fn(output.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
            training_times[folder] = time.time() - start

            # Save model weights
            torch.save(
                model.state_dict(),
                f"saved_models/best_model_{folder}_temporal.pth"
            )

        return training_times

    def test(
        self,
        params: Dict[str, Any],
        device: torch.device
    ) -> Tuple[Dict[str, List[float]], List[float], List[torch.Tensor]]:
        """
        Evaluate the temporal model on each folder's test set.
        Returns:
          - results: dict of metric lists (including train & inference times and AUCPR)
          - inference_times: list of per-folder inference times
          - embeddings_temporal: list of the last-layer embeddings from each folder
        """
        # First, train and collect training times
        training_times = self.train_and_save(
            params,
            device,
            train_epochs=params['n_epochs']
        )

        # Prepare result containers
        results = {
            'test_acc': [],
            'roc_auc': [],
            'aucpr': [],
            'f1_score': [],
            'sensitivity': [],
            'specificity': [],
            'training_time': [],
            'inference_time': []
        }
        embeddings_temporal: List[torch.Tensor] = []

        for folder in params['folders']:
            torch.cuda.empty_cache()

            # Load test temporal data
            _, _, X_test_temporal, _, _, _, _, _, y_test = \
                load_data(device, folder, params["numberOfTimeStep"], params["SG"], params["dataset_name"], params["norm"])

            # Load per-time-step adjacency matrices (kernel + pruning)
            S_temporal = load_temporal_graphs(params['dataset_name'], params['way_to_build_graph'],
                                              folder, params['norm'], params['numberOfTimeStep'], device)

            gnn_params = self.best_params_by_folder[folder]
            gnn_params['seed'] = params['seeds'][int(folder[1:])-1]
            
            # Instantiate and load model
            model = models.TemporalOnlyModel(
                gnn_params=gnn_params,
                S_temporal=S_temporal
            ).to(device)
            state_dict = torch.load(
                f"saved_models/best_model_{folder}_temporal.pth",
                map_location=device
            )
            model.load_state_dict(state_dict)
            model.eval()

            # Record training time
            results['training_time'].append(training_times.get(folder, float('nan')))

            # Inference with timing
            start_inf = time.time()
            with torch.no_grad():
                output_test, temporal_embedding = model(X_test_temporal)
                output_test = output_test.squeeze()
            inf_time = time.time() - start_inf
            results['inference_time'].append(inf_time)
            embeddings_temporal.append(temporal_embedding)

            # Binarize predictions
            pred_test = torch.round(output_test)

            # Compute metrics
            y_np = y_test.cpu().numpy()
            p_np = pred_test.cpu().numpy()
            probs_np = output_test.cpu().numpy()

            results['test_acc'].append(accuracy_score(y_np, p_np))
            results['f1_score'].append(f1_score(y_np, p_np))
            if len(np.unique(y_np)) > 1:
                results['roc_auc'].append(roc_auc_score(y_np, probs_np))
            else:
                results['roc_auc'].append(float('nan'))
            results['aucpr'].append(average_precision_score(y_np, probs_np))

            tn, fp, fn, tp = confusion_matrix(y_np, p_np).ravel()
            results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else float('nan'))
            results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else float('nan'))

        return results, results['inference_time']