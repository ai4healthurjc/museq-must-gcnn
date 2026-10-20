import torch
import pandas as pd
import json
import os
import argparse
from typing import Dict, Any
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

from architectures.MuSTGCNN import MuSTGCNN_utils as utils


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

    params = resolve_params(load_params_from_json(os.path.join(HERE, "load_config_ST.json")), bbdd)
    device = torch.device(params["device"])
    splits = params["folders"]
    graph_method = params["way_to_build_graph"]
    norm = params["norm"]
    norm_static = params.get("norm_static", norm)

    best_results: Dict[str, Any] = {}
    runtime_by_split: Dict[str, float] = {}
    emissions_by_split: Dict[str, float] = {}

    os.makedirs(f"{HERE}/codecarbon_logs/{bbdd}", exist_ok=True)

    i = 0
    for split in splits:
        print(f"\n=== Processing split {split} ===")
        torch.cuda.empty_cache()

        tracker = None if EmissionsTracker is None else EmissionsTracker(
            project_name=f"MuSTGCNN_{bbdd}_{split}",
            output_dir=f"{HERE}/codecarbon_logs/{bbdd}",
            log_level="warning",
            save_to_file=True
        )
        if tracker is not None:
            tracker.start()

        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static, X_val_static, X_test_static,
         y_train, y_val, y_test) = utils.load_data(
            device, split,
            params["numberOfTimeStep"],
            params["SG"],
            bbdd,
            norm, norm_static
        )

        # Static adjacency (dummy si no existe -> temporal-only)
        _sstatic = (f"{PROJECT_ROOT}/graph_estimation/static_data/step2_graphRepresentation/"
                    f"{bbdd}/{graph_method}/{split}/{params['path_S_static']}")
        if os.path.exists(_sstatic):
            S_static = pd.read_csv(_sstatic)
            S_static = torch.tensor(S_static.fillna(0).values, dtype=torch.float32).to(device)
        else:
            _n = params["static_gnn_params"]["embedding_dim"]
            print(f"[S_static] no existe -> dummy identidad {_n}x{_n} (no se usa, use_static=False)")
            S_static = torch.eye(_n, dtype=torch.float32).to(device)

        # ST temporal adjacency
        S_temporal = pd.read_csv(
            f"{PROJECT_ROOT}/graph_estimation/temporal_data/step2_graphRepresentation/"
            f"{bbdd}/{graph_method}/{split}/SpaceTimeGraph_Xtr_{norm}_th_0.975.csv"
        )
        S_temporal = torch.tensor(S_temporal.fillna(0).values, dtype=torch.float32).to(device)

        print(f"Static={S_static.shape}, Temporal={S_temporal.shape}, "
              f"X_train_temporal={X_train_temporal.shape}, y_train={y_train.shape}")

        seed = params["seed"][i]
        i += 1
        optimizer = utils.Optimizer(device=device)

        best_params, runtime = optimizer.optimize_hyperparameters(
            S_temporal=S_temporal,
            S_static=S_static,
            X_train_temporal=X_train_temporal,
            X_train_static=X_train_static,
            X_val_temporal=X_val_temporal,
            X_val_static=X_val_static,
            y_train=y_train,
            y_val=y_val,
            params=params,
            seed=seed
        )

        emissions = tracker.stop() if tracker is not None else None
        emissions_by_split[split] = emissions
        best_results[split] = best_params
        runtime_by_split[split] = runtime

    output_dir = f"{HERE}/hyperparameters/{bbdd}"
    os.makedirs(output_dir, exist_ok=True)

    utils.saveBestHyperparameters(
        best_results,
        os.path.join(output_dir, "MuSTGCNN_hyperparameters.json"),
        params
    )
    with open(os.path.join(output_dir, "MuSTGCNN_runtime.json"), "w") as f:
        json.dump(runtime_by_split, f, indent=4)
    with open(os.path.join(output_dir, "MuSTGCNN_emissions.json"), "w") as f:
        json.dump(emissions_by_split, f, indent=4)

    print(f"Optimization complete. Results saved to {output_dir}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
