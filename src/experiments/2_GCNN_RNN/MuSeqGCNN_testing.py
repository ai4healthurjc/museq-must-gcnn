import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src", "utils")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architectures.MuSeqGCNN import MuSeqGCNN_utils as utils
from architectures.MuSeqGCNN.model import MultimodalGNN


def load_params(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)


def load_graphs(folder: str, params: Dict[str, Any], device: torch.device, bbdd: str):
    way = params["way_to_build_graph"]
    norm = params["norm"]
    norm_static = params.get("norm_static", norm)

    _sstatic = (f"{PROJECT_ROOT}/graph_estimation/static_data/step2_graphRepresentation/"
                f"{bbdd}/{way}/{folder}/{params['path_S_static']}")
    if os.path.exists(_sstatic):
        S_static = pd.read_csv(_sstatic)
        S_static = torch.tensor(S_static.fillna(0).values, dtype=torch.float32).to(device)
    else:
        _n = params["static_gnn_params"]["embedding_dim"]
        print(f"[S_static] no existe -> dummy identidad {_n}x{_n} (no se usa, use_static=False)")
        S_static = torch.eye(_n, dtype=torch.float32).to(device)

    S_temporal_list = []
    for t in range(params["numberOfTimeStep"]):
        df = pd.read_csv(
            f"{PROJECT_ROOT}/graph_estimation/temporal_data/step2_graphRepresentation/"
            f"{bbdd}/{way}/{folder}/adj_X_train_TS_{t}_{norm}_th_0.975.csv"
        )
        S_temporal_list.append(torch.tensor(df.fillna(0).values, dtype=torch.float32).to(device))

    return S_static, S_temporal_list


def slice_temporal(X_seq: torch.Tensor, device: torch.device):
    P, T, F = X_seq.shape
    return [X_seq[:, t, :].reshape(P, F, 1).to(device) for t in range(T)]


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
    p.add_argument("--variant", choices=["gnn_rnn_1", "gnn_rnn_1_transformer"],
                   default="gnn_rnn_1")
    args = p.parse_args()
    bbdd = args.bbdd

    params = resolve_params(load_params(os.path.join(HERE, "load_config_gnn_rnn.json")), bbdd)
    device = torch.device(params["device"])
    best_results = {}
    all_intrinsic_xai: Dict[str, Any] = {}

    hyp_path = f"{HERE}/hyperparameters/{bbdd}/MuSeqGCNN_hyperparameters.json"
    model_dir = f"{HERE}/models/{bbdd}/{args.variant}"
    result_dir = f"{HERE}/results/{bbdd}/{args.variant}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    hyper_all = utils.loadBestHyperparameters(hyp_path)

    for folder, seed in zip(params["folders"], params["seed"]):
        print(f"\n=== {args.variant} | folder {folder} ===")

        X_tr_t, X_val_t, X_te_t, X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te = \
            utils.load_data(device, folder, params["numberOfTimeStep"],
                            params["SG"], bbdd, params["norm"], params.get("norm_static", params["norm"]))

        S_static, S_temporal = load_graphs(folder, params, device, bbdd)
        X_tr_tl  = slice_temporal(X_tr_t, device)
        X_val_tl = slice_temporal(X_val_t, device)
        X_te_tl  = slice_temporal(X_te_t, device)

        hparams = hyper_all[folder]

        tester = utils.TestModel(
            seed=seed,
            model_class=MultimodalGNN,
            device=device,
            best_model_path=f"{model_dir}/model{folder}.pth"
        )

        res, intrinsic_xai = tester.test(
            hyperparams_path=hyp_path,
            split=folder,
            S_temporal_list=S_temporal,
            S_static=S_static,
            X_train_temporal=X_tr_tl,
            X_train_static=X_tr_s,
            y_train=y_tr,
            X_val_temporal=X_val_tl,
            X_val_static=X_val_s,
            y_val=y_val,
            X_test_temporal=X_te_tl,
            X_test_static=X_te_s,
            y_test=y_te,
            n_epochs=hparams["train_params"]["n_epochs"],
            early_stopping_patience=hparams["train_params"]["early_stopping_patience"]
        )

        with open(f"{result_dir}/pred_split_{folder}.json", "w") as pf:
            json.dump({
                "probabilities": res.pop("pred_probs", []),
                "predictions":   res.pop("pred_labels", []),
            }, pf, indent=4)

        best_results[f"{args.variant}_{folder}"] = res
        print(f"→ {res}")

        serializable = {
            key: value.detach().cpu().numpy().tolist()
            for key, value in intrinsic_xai.items()
            if value is not None and hasattr(value, "detach")
        }
        all_intrinsic_xai[folder] = serializable
        for k in ["probability", "static_embedding", "temporal_embedding"]:
            intrinsic_xai.pop(k, None)

    with open(f"{result_dir}/results.json", "w") as rf:
        json.dump(best_results, rf, indent=4)
    with open(f"{result_dir}/xai_intrinsic.json", "w") as xf:
        json.dump(all_intrinsic_xai, xf, indent=4)

    print("\n=== Final Results ===")
    for key, res in best_results.items():
        print(f"{key}: ROC-AUC {res['roc_auc']:.4f}, "
              f"Sens {res['sensitivity']:.4f}, Spec {res['specificity']:.4f}")

    metrics = ["test_acc", "roc_auc", "aucpr", "f1_score",
               "sensitivity", "specificity", "inference_time", "training_time", "mean_weight"]
    print("\n=== Averages Across Folders ===")
    for m in metrics:
        vals = np.array([res.get(m, np.nan) for res in best_results.values()], dtype=float)
        print(f"{m}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")


def clean_up_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
