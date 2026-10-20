import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import argparse
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils", "architectures", "gcnn")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils_temporal as utils


def load_params_from_json(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return json.load(file)


def load_search_space(filepath: str) -> dict:
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
    search_space = load_search_space(os.path.join(HERE, "optuna_search_space.json"))

    folders = params["folders"]
    way_to_build_graph = params["way_to_build_graph"]
    n_trials = params["n_trials"]
    device = torch.device(f"cuda:{params['device_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    SG = params["SG"]
    numberOfTimeStep = params["numberOfTimeStep"]

    best_result_by_split = {}
    optimizer = utils.Optimizer()

    for folder in folders:
        torch.cuda.empty_cache()
        folder = str(folder)
        print(f"===========> ONLY TEMPORAL - FOLDER {folder} ==================")

        X_train_temporal, X_val_temporal, X_test_temporal, _, _, _, y_train, y_val, y_test = utils.load_data(
            device, folder, numberOfTimeStep, SG, bbdd, params["norm"]
        )

        print(f"X_train_temporal shape: {X_train_temporal.shape}")

        S_temporal = utils.load_temporal_graphs(bbdd, way_to_build_graph, folder,
                                                params["norm"], numberOfTimeStep, device)

        seed = params["seeds"][int(folder[1:])-1]
        best_hyperparameters = optimizer.run_optuna(
            folder=folder,
            seed=seed,
            S_temporal=S_temporal,
            S_static=None,
            X_train_temporal=X_train_temporal,
            X_train_static=None,
            X_val_temporal=X_val_temporal,
            X_val_static=None,
            y_train=y_train,
            y_val=y_val,
            params=params,
            device=device,
            search_space=search_space,
            n_trials=n_trials
        )

        print(f"<========== FIN FOLDER {folder} ===============>")
        best_result_by_split[folder] = best_hyperparameters

    output_dir = f"{HERE}/hyperparameters/{bbdd}/{way_to_build_graph}"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = "App1_temporal_ST.json" if SG == "yes" else "App1_temporal.json"
    output_filepath = os.path.join(output_dir, output_filename)
    utils.saveBestHyperparameters(best_result_by_split, output_filepath)
    print(f"Hyperparameters saved to {output_filepath}")


if __name__ == "__main__":
    main()
