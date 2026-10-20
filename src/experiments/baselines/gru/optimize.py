import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import json
import argparse
import contextlib
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

from architectures.gru import gru_utils as utils


@contextlib.contextmanager
def gpu_memory_logger(tag: str = ""):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
    else:
        start_alloc = start_reserved = 0

    yield

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated()
        end_alloc = torch.cuda.memory_allocated()
        print(
            f"[GPU {tag}] "
            f"start={start_alloc/1e9:.2f}GB, "
            f"peak={peak_alloc/1e9:.2f}GB, "
            f"end={end_alloc/1e9:.2f}GB, "
            f"reserved_start={start_reserved/1e9:.2f}GB"
        )


def load_params_from_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        return json.load(file)


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
    n_trials = params["n_trials"]

    best_result_by_split = {}
    optimization_time_by_split = {}
    emissions_by_split = {}

    os.makedirs(f"{HERE}/codecarbon_logs/{bbdd}", exist_ok=True)

    i = 0
    for folder in folders:
        print(f"\n=== Processing folder {folder} ===")
        torch.cuda.empty_cache()

        tracker = None if EmissionsTracker is None else EmissionsTracker(
            project_name=f"gru_hyperopt_{bbdd}_{folder}",
            output_dir=f"{HERE}/codecarbon_logs/{bbdd}",
            measure_power_secs=60,
            log_level="error",
        )
        if tracker is not None:
            tracker.start()

        (X_train_temporal, X_val_temporal, X_test_temporal,
         _, _, _,
         y_train, y_val, y_test) = utils.load_data(
            device=device,
            split=folder,
            numberOfTimeStep=params["numberOfTimeStep"],
            SG=params["SG"],
            dataset_name=bbdd,
            norm=params["norm"]
        )

        print("\n=== DATA SHAPES ===")
        print(f"Temporal training shape: {X_train_temporal.shape}")
        print(f"Validation shape: {X_val_temporal.shape}")
        print(f"Training labels shape: {y_train.shape}")

        seed = params['train_params']["seeds"][i]
        i += 1
        optimizer = utils.Optimizer(device=device, seed=seed)

        with gpu_memory_logger(tag=f"{folder}_seed{seed}"):
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

        emissions = tracker.stop() if tracker is not None else None
        emissions_by_split[folder] = emissions
        print(f"Emissions for folder {folder}: {emissions:.6f} kg CO₂eq")

    # Save results
    output_dir = f"{HERE}/hyperparameters/{bbdd}"
    os.makedirs(output_dir, exist_ok=True)

    hyperparam_path = os.path.join(output_dir, "best_hyperparams.json")
    utils.saveBestHyperparameters(best_result_by_split, hyperparam_path, params)

    opt_time_path = os.path.join(output_dir, "opt_time.json")
    with open(opt_time_path, "w") as f:
        json.dump(optimization_time_by_split, f, indent=4)

    emissions_path = os.path.join(output_dir, "emissions_by_split.json")
    with open(emissions_path, "w") as f:
        json.dump(emissions_by_split, f, indent=4)

    print(f"Hyperparameters saved to {hyperparam_path}")
    print(f"Optimization times saved to {opt_time_path}")
    print(f"Emissions saved to {emissions_path}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
