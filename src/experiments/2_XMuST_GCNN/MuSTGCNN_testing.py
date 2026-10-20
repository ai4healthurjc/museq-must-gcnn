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
    os.path.join(os.path.dirname(__file__), "../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architectures.MuSTGCNN import MuSTGCNN_utils as utils
from architectures.MuSTGCNN.model import MultimodalGNN_ST as MultimodalGNN


def load_params_from_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)


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
    graph_method = params["way_to_build_graph"]
    norm = params["norm"]
    norm_static = params.get("norm_static", norm)

    all_results = {}
    all_intrinsic_xai: Dict[str, Any] = {}

    hyperparams_path = f"{HERE}/hyperparameters/{bbdd}/MuSTGCNN_hyperparameters.json"
    model_dir = f"{HERE}/models/{bbdd}/MuSTGCNN"
    result_dir = f"{HERE}/results/{bbdd}/MuSTGCNN"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    for folder, seed in zip(params["folders"], params["seed"]):
        print(f"\n=== folder {folder} ===")

        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static, X_val_static, X_test_static,
         y_train, y_val, y_test) = utils.load_data(
            device, folder,
            params["numberOfTimeStep"],
            params["SG"],
            bbdd,
            norm, norm_static
        )

        # Static adjacency (dummy si no existe -> temporal-only)
        _sstatic = (f"{PROJECT_ROOT}/graph_estimation/static_data/step2_graphRepresentation/"
                    f"{bbdd}/{graph_method}/{folder}/{params['path_S_static']}")
        if os.path.exists(_sstatic):
            S_static = pd.read_csv(_sstatic).values
            S_static = torch.tensor(S_static, dtype=torch.float32).to(device)
        else:
            _n = params["static_gnn_params"]["embedding_dim"]
            print(f"[S_static] no existe -> dummy identidad {_n}x{_n} (no se usa, use_static=False)")
            S_static = torch.eye(_n, dtype=torch.float32).to(device)

        # ST temporal adjacency
        S_temporal = pd.read_csv(
            f"{PROJECT_ROOT}/graph_estimation/temporal_data/step2_graphRepresentation/"
            f"{bbdd}/{graph_method}/{folder}/SpaceTimeGraph_Xtr_{norm}_th_0.975.csv"
        ).values
        S_temporal = torch.tensor(S_temporal, dtype=torch.float32).to(device)

        best_hps = utils.loadBestHyperparameters(hyperparams_path)[folder]

        tester = utils.TestModel(
            seed=seed,
            model_class=MultimodalGNN,
            device=device,
            best_model_path=f"{model_dir}/model_ST_{folder}.pth"
        )

        results, intrinsic_xai = tester.test(
            hyperparams_path=hyperparams_path,
            split=folder,
            S_temporal_list=S_temporal,
            S_static=S_static,
            X_train_temporal=X_train_temporal,
            X_train_static=X_train_static,
            y_train=y_train,
            X_val_temporal=X_val_temporal,
            X_val_static=X_val_static,
            y_val=y_val,
            X_test_temporal=X_test_temporal,
            X_test_static=X_test_static,
            y_test=y_test,
            n_epochs=best_hps["train_params"]["n_epochs"],
            early_stopping_patience=best_hps["train_params"]["early_stopping_patience"]
        )

        all_results[folder] = results
        print(f"Split {folder} results:", results)

        serializable = {
            key: value.detach().cpu().numpy().tolist()
            for key, value in intrinsic_xai.items()
            if value is not None and hasattr(value, "detach")
        }
        all_intrinsic_xai[folder] = serializable
        for k in ["probability", "static_embedding", "temporal_embedding"]:
            intrinsic_xai.pop(k, None)

    with open(os.path.join(result_dir, "results.json"), "w") as mf:
        json.dump(all_results, mf, indent=4)
    with open(os.path.join(result_dir, "MuSTGCNN_xai_intrinsic.json"), "w") as xf:
        json.dump(all_intrinsic_xai, xf, indent=4)

    print(f"\nResults saved to {result_dir}")

    metrics = ["test_acc", "roc_auc", "aucpr", "f1_score",
               "sensitivity", "specificity", "inference_time", "training_time"]
    df = pd.DataFrame.from_dict(all_results, orient="index")
    summary = df[metrics].agg(["mean", "std"])
    print("\n=== Summary of all splits ===")
    print(summary.round(4))


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
