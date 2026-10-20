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
import pandas as pd
import json
import numpy as np
import argparse
import os
from typing import Dict, Any, List
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architectures.gru import gru_utils as utils
from architectures.gru.model import GRUTemporalModel

import matplotlib.pyplot as plt


def load_params_from_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        return json.load(file)


def plot_and_save_curves(history: Dict[str, List[float]], out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)

    import csv
    csv_path = os.path.join(out_dir, f"{tag}_history.csv")
    keys = ["train_loss", "val_loss", "val_roc_auc"]
    max_len = max(len(history.get(k, [])) for k in keys)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(max_len):
            row = [i + 1]
            for k in keys:
                vals = history.get(k, [])
                row.append(vals[i] if i < len(vals) else "")
            writer.writerow(row)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(8, 5))
    if len(history.get("train_loss", [])) > 0:
        plt.plot(epochs, history["train_loss"], label="Train Loss")
    if len(history.get("val_loss", [])) > 0:
        plt.plot(epochs, history["val_loss"], label="Val Loss", linestyle="--")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"{tag} - Pérdidas (Train/Val)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(out_dir, f"{tag}_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    if len(history.get("val_roc_auc", [])) > 0:
        plt.plot(epochs, history["val_roc_auc"], label="Val ROC-AUC")
    plt.xlabel("Época")
    plt.ylabel("ROC-AUC (Val)")
    plt.title(f"{tag} - ROC-AUC (Validación)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(out_dir, f"{tag}_roc_auc.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return {"csv": csv_path, "loss_png": loss_path, "roc_png": roc_path}


def resolve_params(cfg, bbdd):
    """Fusiona parametros comunes + bloque especifico de la BBDD."""
    if "datasets" not in cfg:
        return cfg
    if bbdd not in cfg["datasets"]:
        raise KeyError(f"BBDD '{bbdd}' no esta en load_config.json "
                       f"(datasets: {list(cfg['datasets'])})")
    p = {k: v for k, v in cfg.items() if k != "datasets"}
    p.update(cfg["datasets"][bbdd])
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbdd", required=True, help="Nombre de la BBDD (e.g. AMR)")
    args = p.parse_args()
    bbdd = args.bbdd

    params = resolve_params(load_params_from_json(os.path.join(HERE, "load_config.json")), bbdd)
    device = torch.device(params["device"])
    folders = params["folders"]
    best_results = {}

    results_dir = f"{HERE}/results/{bbdd}"
    plots_dir = os.path.join(results_dir, "plots")
    models_dir = f"{HERE}/models/{bbdd}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    seeds = params['train_params']['seeds']
    for folder, seed in zip(folders, seeds):
        print(f"\n=== Processing folder {folder} ===")

        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static, X_val_static, X_test_static,
         y_train, y_val, y_test) = utils.load_data(
            device, folder,
            params["numberOfTimeStep"],
            params["SG"],
            bbdd,
            params["norm"]
        )

        hyperparams_path = f"{HERE}/hyperparameters/{bbdd}/best_hyperparams.json"
        all_hyperparams = utils.loadBestHyperparameters(hyperparams_path)
        split_hyperparams = all_hyperparams[folder]

        test_model = utils.TestModel(
            seed=seed,
            model_class=GRUTemporalModel,
            device=device,
            best_model_path=f"{models_dir}/model{folder}.pth"
        )

        results = test_model.test(
            hyperparams_path=hyperparams_path,
            split=folder,
            X_train_temporal=X_train_temporal,
            y_train=y_train,
            X_val_temporal=X_val_temporal,
            y_val=y_val,
            X_test_temporal=X_test_temporal,
            y_test=y_test,
            n_epochs=split_hyperparams["train_params"]["n_epochs"],
            early_stopping_patience=split_hyperparams["train_params"]["early_stopping_patience"]
        )

        history = results.get("history", None)
        if history is not None:
            tag = f"folder_{folder}"
            saved = plot_and_save_curves(history, plots_dir, tag)
            print(f"[{folder}] Curvas guardadas: {saved['csv']}")
        else:
            print(f"[{folder}] No se encontró 'history' en results.")

        results.pop("pred_probs", None)
        results.pop("pred_labels", None)

        best_results[str(folder)] = results
        print(f"Finished folder {folder}.")

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(best_results, f, indent=4)

    print("\n=== Final Results ===")
    for folder, res in best_results.items():
        print(f"Folder {folder}: ROC-AUC: {res['roc_auc']:.4f}, "
              f"Sensitivity: {res['sensitivity']:.4f}, "
              f"Specificity: {res['specificity']:.4f}")

    metrics = ["test_acc", "roc_auc", "aucpr", "f1_score",
               "sensitivity", "specificity", "inference_time", "training_time"]

    arrs = {m: [] for m in metrics}
    for res in best_results.values():
        for m in metrics:
            arrs[m].append(res.get(m, 0))

    print("\n=== Average Results Across Folders ===")
    for m in metrics:
        values = np.array(arrs[m])
        print(f"{m}: {values.mean():.4f} ± {values.std():.4f}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
