import torch
import pandas as pd
import json
import numpy as np
import os
import argparse
from typing import Dict, Any
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architectures.mlp import mlp_utils as utils
from architectures.mlp.model import MLPStaticModel


def load_params_from_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        return json.load(file)


def resolve_params(cfg, bbdd):
    """Fusiona parametros comunes + bloque especifico de la BBDD."""
    if "datasets" not in cfg:
        return cfg
    if bbdd not in cfg["datasets"]:
        raise KeyError(f"BBDD '{bbdd}' no esta en config (datasets: {list(cfg['datasets'])})")
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

    results_dir = f"{HERE}/results/{bbdd}/mlp"
    models_dir = f"{HERE}/models/{bbdd}"
    preds_dir = os.path.join(results_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    i = 0
    for folder in folders:
        print(f"\n=== Processing folder {folder} ===")

        (_, _, _,
         X_train_static, X_val_static, X_test_static,
         y_train, y_val, y_test) = utils.load_data(
            device, folder,
            params["numberOfTimeStep"],
            params["SG"],
            bbdd,
            params["norm"]
        )

        hyperparams_path = f"{HERE}/hyperparameters/{bbdd}/mlp/best_hyperparams.json"
        all_hyperparams = utils.loadBestHyperparameters(hyperparams_path)
        split_hyperparams = all_hyperparams[folder]

        seed = params["train_params"]["seeds"][i]
        i += 1

        test_model = utils.TestModel(
            seed=seed,
            folder=folder,
            model_class=MLPStaticModel,
            device=device,
            best_model_path=f"{models_dir}/model_{folder}.pth"
        )

        results = test_model.test(
            hyperparams_path=hyperparams_path,
            split=folder,
            X_train_static=X_train_static,
            y_train=y_train,
            X_val_static=X_val_static,
            y_val=y_val,
            X_test_static=X_test_static,
            y_test=y_test,
            n_epochs=split_hyperparams["train_params"]["n_epochs"],
            early_stopping_patience=split_hyperparams["train_params"]["early_stopping_patience"]
        )

        preds_path = os.path.join(preds_dir, f"pred_split_{folder}.json")
        with open(preds_path, "w") as pf:
            json.dump({
                "probabilities": results["pred_probs"],
                "predictions": results["pred_labels"]
            }, pf, indent=4)

        for k in ["pred_probs", "pred_labels"]:
            results.pop(k, None)

        best_results[folder] = results
        print(f"Results for folder {folder}:", results)

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(best_results, f, indent=4)

    print("\n=== Final Results ===")
    for folder, res in best_results.items():
        print(f"Folder {folder}: ROC-AUC: {res['roc_auc']:.4f}, "
              f"Sensitivity: {res['sensitivity']:.4f}, "
              f"Specificity: {res['specificity']:.4f}")

    metrics = ["test_acc", "roc_auc", "aucpr", "f1_score",
               "sensitivity", "specificity", "inference_time", "training_time"]
    print("\n=== Average Results Across Folders ===")
    for k in metrics:
        arr = np.array([res[k] for res in best_results.values()])
        print(f"{k}: {arr.mean():.4f} ± {arr.std():.4f}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()