import torch
import pandas as pd
import json
import os
import argparse
from typing import Dict, Any, List
try:
    from codecarbon import EmissionsTracker
except ImportError:  # optional dependency: energy tracking is disabled if absent
    EmissionsTracker = None
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architectures.MuSeqGCNN import MuSeqGCNN_utils as utils


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

    params = resolve_params(load_params_from_json(os.path.join(HERE, "load_config_gnn_rnn.json")), bbdd)
    device = torch.device(params["device"])
    folders = params["folders"]
    way_to_build_graph = params["way_to_build_graph"]
    n_trials = params["n_trials"]
    norm = params["norm"]
    norm_static = params.get("norm_static", norm)

    best_result_by_split = {}
    optimization_time_by_split = {}
    emissions_by_split = {}

    i = 0
    for folder in folders:
        print(f"\n=== Processing folder {folder} ===")
        torch.cuda.empty_cache()

        os.makedirs(f"{HERE}/codecarbon_logs/{bbdd}", exist_ok=True)
        tracker = None if EmissionsTracker is None else EmissionsTracker(
            project_name=f"MuSeqGCNN_hyperparam_opt_{bbdd}_{folder}",
            output_dir=f"{HERE}/codecarbon_logs/{bbdd}",
            save_to_file=True,
            log_level="error"
        )
        if tracker is not None:
            tracker.start()

        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static, X_val_static, X_test_static,
         y_train, y_val, y_test) = utils.load_data(
            device, folder,
            params["numberOfTimeStep"],
            params["SG"],
            bbdd,
            norm, norm_static
        )

        # Load static graph (dummy si no existe -> temporal-only con use_static=False)
        _sstatic_path = (f"{PROJECT_ROOT}/graph_estimation/static_data/step2_graphRepresentation/"
                         f"{bbdd}/{way_to_build_graph}/{folder}/{params['path_S_static']}")
        if os.path.exists(_sstatic_path):
            S_static = pd.read_csv(_sstatic_path)
            S_static = torch.tensor(S_static.values, dtype=torch.float32).to(device)
        else:
            _n = params["static_gnn_params"]["embedding_dim"]
            print(f"[S_static] no existe -> dummy identidad {_n}x{_n} (no se usa, use_static=False)")
            S_static = torch.eye(_n, dtype=torch.float32).to(device)

        # Load temporal graphs (one per timestep)
        S_temporal_list = []
        for nt in range(params["numberOfTimeStep"]):
            S_t = pd.read_csv(
                f"{PROJECT_ROOT}/graph_estimation/temporal_data/step2_graphRepresentation/"
                f"{bbdd}/{way_to_build_graph}/{folder}/adj_X_train_TS_{nt}_{norm}_th_0.975.csv"
            )
            S_temporal_list.append(torch.tensor(S_t.fillna(0).values, dtype=torch.float32).to(device))

        # Reshape temporal data into list format
        X_train_temporal_list = []
        X_val_temporal_list = []
        for t in range(params["numberOfTimeStep"]):
            X_train_temporal_list.append(
                X_train_temporal[:, t, :].reshape(X_train_temporal.shape[0], X_train_temporal.shape[2], 1)
            )
            X_val_temporal_list.append(
                X_val_temporal[:, t, :].reshape(X_val_temporal.shape[0], X_val_temporal.shape[2], 1)
            )

        print(f"Static adj: {S_static.shape}, Temporal adj: {len(S_temporal_list)}x{S_temporal_list[0].shape}")
        print(f"X_train_temporal_list: {len(X_train_temporal_list)}x{X_train_temporal_list[0].shape}")

        optimizer = utils.Optimizer(device=device)
        seed = params["seed"][i]
        i += 1

        best_hyperparameters, optimization_time = optimizer.optimize_hyperparameters(
            S_temporal_list=S_temporal_list,
            S_static=S_static,
            X_train_temporal=X_train_temporal_list,
            X_train_static=X_train_static,
            X_val_temporal=X_val_temporal_list,
            X_val_static=X_val_static,
            y_train=y_train,
            y_val=y_val,
            params=params,
            seed=seed,
            n_trials=n_trials
        )

        best_result_by_split[folder] = best_hyperparameters
        optimization_time_by_split[folder] = optimization_time

        emissions = tracker.stop() if tracker is not None else None
        emissions_by_split[folder] = emissions
        print(f"CO2 emissions for folder {folder}: {emissions} kg")

    output_dir = f"{HERE}/hyperparameters/{bbdd}"
    os.makedirs(output_dir, exist_ok=True)

    utils.saveBestHyperparameters(
        best_result_by_split,
        os.path.join(output_dir, "MuSeqGCNN_hyperparameters.json"),
        params
    )
    with open(os.path.join(output_dir, "MuSeqGCNN_opt_time.json"), "w") as f:
        json.dump(optimization_time_by_split, f, indent=4)
    with open(os.path.join(output_dir, "MuSeqGCNN_emissions.json"), "w") as f:
        json.dump(emissions_by_split, f, indent=4)

    print(f"Results saved to {output_dir}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
