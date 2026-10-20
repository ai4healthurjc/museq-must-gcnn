import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, average_precision_score  
import optuna    
import torch
import json
import torch.nn as nn
import time
from .model import LogisticStaticModel
from typing import Dict, Any, List
import random
import sys
import os
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")

###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################

def saveBestHyperparameters(best_hyperparameters, filename, original_params):
    import copy
    saved_params = copy.deepcopy(original_params)

    for split in best_hyperparameters:
        trial_params = best_hyperparameters[split]

        train_params = {
            "learning_rate": trial_params["learning_rate"],
            "decay": trial_params["decay"],
            "early_stopping_patience": original_params["train_params"]["early_stopping_patience"]
        }

        saved_params[split] = {
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

    

class Optimizer:
    """
    A class for training models and performing hyperparameter optimization.
    """
    def __init__(self, seed: int, device, loss_function=nn.BCELoss(), storage=None, study_name=None):
        self.device = device
        self.seed = seed
        self.loss_function = loss_function
        self.best_model_path = "best_multimodal_model.pth"
        self.storage = storage
        self.study_name = study_name

    def train(self, model, X_temporal, y, optimizer):
        """Single training step for temporal-only Mamba model"""
        model.train()
        optimizer.zero_grad()
        output = model(X_temporal).squeeze()
        loss = self.loss_function(output, y)
        loss.backward()
        optimizer.step()
        return loss


    def objective(self, trial, X_train_static, X_val_static, y_train, y_val, params):
                
        lr = trial.suggest_float('learning_rate', 
                               low=params['train_params']['learning_rate'][0],
                               high=params['train_params']['learning_rate'][1], 
                               log=True)
        decay = trial.suggest_float('decay', 
                                  low=params['train_params']['decay'][0],
                                  high=params['train_params']['decay'][1], 
                                  log=True)



        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


        model = LogisticStaticModel(input_dim=params["static_input_dim"], seq_len=1).to(self.device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        best_roc_auc = -1
        early_stop_counter = 0

        for epoch in range(params['train_params']['n_epochs']):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_static).squeeze()
            loss = self.loss_function(pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_static).squeeze()
                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_pred.cpu().numpy())

                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    early_stop_counter = 0
                    torch.save(model.state_dict(), self.best_model_path)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= params['train_params']['early_stopping_patience']:
                    break

        return best_roc_auc


    
    def optimize_hyperparameters(self, X_train_temporal, X_val_temporal,
                                y_train, y_val, params, n_trials=100):

        import time
        sampler = optuna.samplers.TPESampler(seed=self.seed)
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
                trial, X_train_temporal, X_val_temporal,
                y_train, y_val, params
            ),
            n_trials=params['n_trials']
        )
        
        optimization_time = time.time() - start_time
        return study.best_params, optimization_time



class TestModel:
    def __init__(self, seed, folder, model_class, device, loss_function=nn.BCELoss(), best_model_path=None):
        self.seed = seed
        self.folder = folder
        self.model_class = model_class
        self.device = device
        self.loss_function = loss_function

        if best_model_path is None:
            self.best_model_path = f"./models/model_{folder}.pth"
        else:
            self.best_model_path = best_model_path

    def load_hyperparameters(self, hyperparams_path, split):
        with open(hyperparams_path, 'r') as file:
            all_hyperparams = json.load(file)
            params_split = all_hyperparams[split]
            params_split['static_input_dim'] = all_hyperparams['static_input_dim']
        return params_split

    def train_model(self, hyperparams,
                    X_train_static, y_train,
                    X_val_static, y_val,
                    n_epochs, early_stopping_patience):
        import time

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        input_dim = hyperparams["static_input_dim"]

        model = self.model_class(input_dim=input_dim, seq_len=1).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams["train_params"]["learning_rate"],
            weight_decay=hyperparams["train_params"]["decay"]
        )

        best_roc_auc = -1
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_static).squeeze()
            loss = self.loss_function(output, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_static).squeeze()
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
        return model, training_time

    def evaluate(self, model, X_test_static, y_test):
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            pred_probs = model(X_test_static)
            end_time = time.time()

            inference_time = end_time - start_time
            pred_probs = pred_probs.squeeze()
            pred = torch.round(pred_probs).view(-1)

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
            "pred_probs": pred_probs.cpu().numpy().tolist(),
            "pred_labels": pred.cpu().numpy().tolist()
        }

    def test(self, hyperparams_path, split,
             X_train_static, y_train,
             X_val_static, y_val,
             X_test_static, y_test,
             n_epochs, early_stopping_patience):

        hyperparams = self.load_hyperparameters(hyperparams_path, split)

        model, training_time = self.train_model(
            hyperparams,
            X_train_static, y_train,
            X_val_static, y_val,
            n_epochs, early_stopping_patience
        )

        results = self.evaluate(model, X_test_static, y_test)
        results["training_time"] = training_time
        return results
