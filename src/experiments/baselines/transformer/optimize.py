import torch
import pandas as pd
import json
import os
import argparse
from typing import Dict, Any
import sys
try:
    from codecarbon import EmissionsTracker
except ImportError:  # optional dependency: energy tracking is disabled if absent
    EmissionsTracker = None

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architectures.transformer import transformer_utils as utils


def load_params_from_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        return json.load(file)


def resolve_params(cfg, bbdd):
    """Fusiona parametros comunes + bloque especifico de la BBDD."""
    if "datasets" not in cfg:
        return cfg
    if bbdd not in cfg["datasets"]:
        raise KeyError(f"BBDD '{bbdd}' no esta en load_config.json (datasets: {list(cfg['datasets'])})")
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
    n_trials = params["n_trials"]

    best_result_by_split = {}
    optimization_time_by_split = {}

    os.makedirs(f"{HERE}/codecarbon_logs/{bbdd}", exist_ok=True)
    tracker = None if EmissionsTracker is None else EmissionsTracker(
        project_name=f"transformer_hyperparam_search_{bbdd}",
        output_dir=f"{HERE}/codecarbon_logs/{bbdd}",
        save_to_file=True
    )
    if tracker is not None:
            tracker.start()

    i = 0
    for folder in folders:
        print(f"\n=== Processing folder {folder} ===")
        torch.cuda.empty_cache()

        (X_train_temporal, X_val_temporal, X_test_temporal,
         _, _, _,
         y_train, y_val, y_test) = utils.load_data(
            device, folder,
            params["numberOfTimeStep"],
            params["SG"],
            bbdd,
            params["norm"]
        )

        print(f"Temporal training shape: {X_train_temporal.shape}")
        print(f"Labels shape: {y_train.shape}")

        seed = params["train_params"]["seeds"][i]
        i += 1
        optimizer = utils.Optimizer(device=device, seed=seed)

        best_hyperparameters, optimization_time = optimizer.optimize_hyperparameters(
            X_train_temporal=X_train_temporal,
            X_val_temporal=X_val_temporal,
            y_train=y_train,
            y_val=y_val,
            params=params,
            n_trials=n_trials
        )

        best_result_by_split[folder] = best_hyperparameters
        optimization_time_by_split[folder] = optimization_time

    output_dir = f"{HERE}/hyperparameters/{bbdd}"
    os.makedirs(output_dir, exist_ok=True)

    hyperparam_path = os.path.join(output_dir, "best_hyperparams.json")
    utils.saveBestHyperparameters(best_result_by_split, hyperparam_path, params)

    with open(os.path.join(output_dir, "opt_time.json"), "w") as f:
        json.dump(optimization_time_by_split, f, indent=4)

    emissions = tracker.stop() if tracker is not None else None
    print(f"\nEmissions: {emissions} kg CO₂eq")
    print(f"Hyperparameters saved to {hyperparam_path}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
