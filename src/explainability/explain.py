#!/usr/bin/env python3
"""Adapted GNNExplainer for MuSeq-GCNN and MuST-GCNN (Section II-E of the paper).

For every test patient, a temporal mask (F x T) and a static mask (D) are
optimized with the trained model frozen and in eval mode, minimizing

    BCE(y_bar, y_hat(m_t, m_s)) + lambda_1 * size + lambda_2 * entropy,

where y_bar is the label PREDICTED by the original model. Patient-level masks
are then averaged within each ground-truth outcome group (equal number of
patients per group) to obtain population-level explanations.

Usage (from the repository root):
    python src/explainability/explain.py --bbdd AMR --model MuSeqGCNN
    python src/explainability/explain.py --bbdd physionet2012 --model MuSTGCNN --folders s1 s2

Outputs, for every split and for the average over the requested splits:
    explainability_outputs/<bbdd>/<model>/<split | all>/mean_{temporal,static}_{pos,neg}.npy
The temporal masks are flattened with index t*F + f (reshape to (T, F)).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "utils"))

EXPERIMENTS = {
    "MuSeqGCNN": {"dir": "2_GCNN_RNN", "config": "load_config_gnn_rnn.json"},
    "MuSTGCNN": {"dir": "2_XMuST_GCNN", "config": "load_config_ST.json"},
}
GRAPH_THRESHOLD = "0.975"


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
class MultimodalFeatureMaskExplainer(nn.Module):
    """Adapted GNNExplainer with one temporal mask (F*T, index t*F+f) and one
    static mask (D). The trained model is frozen and kept in eval mode."""
    def __init__(self, model: nn.Module, temporal_dim: int, static_dim: int):
        super().__init__()
        self.model = model
        self.model.eval()                                   # no dropout while explaining
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.temporal_mask = nn.Parameter(torch.randn(temporal_dim))
        self.static_mask = nn.Parameter(torch.randn(static_dim, 1))
        self.loss_fn = nn.BCELoss()

    def model_probability(self, x_temporal, x_static) -> torch.Tensor:
        out = self.model(x_temporal, x_static)
        if isinstance(out, dict):
            out = out["probability"]
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(-1)

    def forward(self, x_temporal, x_static: torch.Tensor) -> torch.Tensor:
        sm = torch.sigmoid(self.static_mask)
        x_stat_masked = x_static * sm
        if x_stat_masked.dim() == 2:
            x_stat_masked = x_stat_masked.unsqueeze(0)
        if isinstance(x_temporal, list):                    # MuSeq-GCNN: one tensor per time step
            tm = torch.sigmoid(self.temporal_mask).view(len(x_temporal), -1)
            x_temp_masked = [x * tm[t].view(1, -1, 1) for t, x in enumerate(x_temporal)]
        else:                                               # MuST-GCNN: (1, F*T, 1)
            tm = torch.sigmoid(self.temporal_mask).view(1, -1, 1)
            x_temp_masked = x_temporal * tm
        return self.model_probability(x_temp_masked, x_stat_masked)

    def compute_loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
                     entropy_weight=0.1, l1_weight=0.005) -> torch.Tensor:
        """BCE against the model's own prediction + size (L1) and entropy terms."""
        pred_loss = self.loss_fn(y_pred, y_target)
        t_sig = torch.sigmoid(self.temporal_mask)
        s_sig = torch.sigmoid(self.static_mask)
        ent_t = -(t_sig * torch.log(t_sig + 1e-8) + (1 - t_sig) * torch.log(1 - t_sig + 1e-8)).mean()
        ent_s = -(s_sig * torch.log(s_sig + 1e-8) + (1 - s_sig) * torch.log(1 - s_sig + 1e-8)).mean()
        l1 = t_sig.mean() + s_sig.mean()
        return pred_loss + entropy_weight * (ent_t + ent_s) + l1_weight * l1


def explain_instance(explainer, x_temp, x_static, y_target, optimizer, epochs):
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = explainer(x_temp, x_static)
        loss = explainer.compute_loss(pred, y_target)
        loss.backward()
        optimizer.step()
    return (explainer.temporal_mask.sigmoid().cpu().detach().numpy(),
            explainer.static_mask.sigmoid().cpu().detach().numpy())


def batch_explain(model, X_temp, X_stat, y, params, cfg=None, device=None):
    """Patient-level masks (target = label predicted by the model), averaged
    within each ground-truth outcome group over an equal number of patients."""
    if device is None and isinstance(cfg, torch.device):
        cfg, device = None, cfg
    T = params["numberOfTimeStep"]
    as_list = isinstance(X_temp, torch.Tensor) and X_temp.dim() == 3 and X_temp.shape[1] == T \
        and not (cfg is not None and cfg.get("SG", False))
    pos_idx = (y.cpu().numpy().flatten() == 1).nonzero()[0].tolist()
    neg_idx = (y.cpu().numpy().flatten() == 0).nonzero()[0].tolist()
    n = min(len(pos_idx), len(neg_idx))
    mpc = params.get("max_per_class") or 0
    if mpc > 0:
        n = min(n, mpc)
    selected = {"pos": random.sample(pos_idx, n), "neg": random.sample(neg_idx, n)}
    out = {}
    for label, idxs in selected.items():
        masks_t, masks_s = [], []
        for i in tqdm(idxs, desc=f"Explaining {label} samples"):
            if as_list:
                x_input = [X_temp[i, t].unsqueeze(0).unsqueeze(-1).to(device) for t in range(T)]
            else:
                x_input = X_temp[i].unsqueeze(0).to(device)
            x_s = X_stat[i].to(device)
            explainer = MultimodalFeatureMaskExplainer(
                model, temporal_dim=X_temp[i].numel(), static_dim=X_stat.shape[1]).to(device)
            with torch.no_grad():
                x_s_in = x_s.unsqueeze(0) if x_s.dim() == 2 else x_s
                p0 = explainer.model_probability(x_input, x_s_in)
            y_target = (p0 >= params.get("pred_threshold", 0.5)).float()
            optimizer = torch.optim.Adam([explainer.temporal_mask, explainer.static_mask],
                                         lr=params["explainer_lr"])
            tm, sm = explain_instance(explainer, x_input, x_s, y_target, optimizer,
                                      params["explainer_epochs"])
            masks_t.append(tm)
            masks_s.append(sm)
        out[label] = (np.stack(masks_t).mean(axis=0), np.stack(masks_s).mean(axis=0))
    return out["pos"][0], out["neg"][0], out["pos"][1], out["neg"][1]


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_params(cfg, bbdd):
    p = {k: v for k, v in cfg.items() if k != "datasets"}
    p.update(cfg["datasets"][bbdd])
    return p


def load_graphs(model_name, bbdd, way, folder, norm, T, path_S_static, D, device):
    static_path = (f"{PROJECT_ROOT}/graph_estimation/static_data/step2_graphRepresentation/"
                   f"{bbdd}/{way}/{folder}/{path_S_static}")
    if os.path.exists(static_path):
        S_static = torch.tensor(pd.read_csv(static_path).fillna(0).values,
                                dtype=torch.float32, device=device)
    else:
        S_static = torch.eye(D, dtype=torch.float32, device=device)
    base = f"{PROJECT_ROOT}/graph_estimation/temporal_data/step2_graphRepresentation/{bbdd}/{way}/{folder}"
    if model_name == "MuSeqGCNN":
        S_temporal = [torch.tensor(pd.read_csv(f"{base}/adj_X_train_TS_{t}_{norm}_th_{GRAPH_THRESHOLD}.csv")
                                   .fillna(0).values, dtype=torch.float32, device=device)
                      for t in range(T)]
    else:
        S_temporal = torch.tensor(pd.read_csv(f"{base}/SpaceTimeGraph_Xtr_{norm}_th_{GRAPH_THRESHOLD}.csv")
                                  .fillna(0).values, dtype=torch.float32, device=device)
    return S_temporal, S_static


def build_model(model_name, hp, S_temporal, S_static, seed, use_static, device):
    hp["temporal_gnn_params"]["seed"] = seed
    hp["static_gnn_params"]["seed"] = seed
    if model_name == "MuSeqGCNN":
        from architectures.MuSeqGCNN.model import MultimodalGNN
        model = MultimodalGNN(gnn_params_temporal=hp["temporal_gnn_params"],
                              gnn_params_static=hp["static_gnn_params"],
                              rnn_params=hp["rnn_params"],
                              S_temporal=S_temporal, S_static=S_static,
                              use_static=use_static)
    else:
        from architectures.MuSTGCNN.model import MultimodalGNN_ST
        model = MultimodalGNN_ST(gnn_params_temporal=hp["temporal_gnn_params"],
                                 gnn_params_static=hp["static_gnn_params"],
                                 S_temporal=S_temporal, S_static=S_static,
                                 use_static=use_static)
    return model.to(device)


def weights_path(model_name, exp_dir, bbdd, folder):
    if model_name == "MuSeqGCNN":
        return f"{exp_dir}/models/{bbdd}/gnn_rnn_1/model{folder}.pth"
    return f"{exp_dir}/models/{bbdd}/MuSTGCNN/model_ST_{folder}.pth"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbdd", required=True)
    ap.add_argument("--model", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--folders", nargs="*", default=None, help="splits to explain (default: all)")
    ap.add_argument("--explainer_epochs", type=int, default=300)
    ap.add_argument("--explainer_lr", type=float, default=0.01)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = all positives and as many negatives")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "explainability_outputs"))
    args = ap.parse_args()

    exp = EXPERIMENTS[args.model]
    exp_dir = os.path.join(PROJECT_ROOT, "src", "experiments", exp["dir"])
    params = resolve_params(json.load(open(os.path.join(exp_dir, exp["config"]))), args.bbdd)
    if args.model == "MuSeqGCNN":
        from architectures.MuSeqGCNN import MuSeqGCNN_utils as utils
        hp_file = f"{exp_dir}/hyperparameters/{args.bbdd}/MuSeqGCNN_hyperparameters.json"
    else:
        from architectures.MuSTGCNN import MuSTGCNN_utils as utils
        hp_file = f"{exp_dir}/hyperparameters/{args.bbdd}/MuSTGCNN_hyperparameters.json"
    hyper_all = json.load(open(hp_file))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    T = params["numberOfTimeStep"]
    norm = params["norm"]
    norm_static = params.get("norm_static", norm)
    folders = args.folders or params["folders"]
    seeds = dict(zip(params["folders"], params["seed"]))
    explainer_params = {"numberOfTimeStep": T, "explainer_lr": args.explainer_lr,
                        "explainer_epochs": args.explainer_epochs,
                        "max_per_class": args.max_per_class}

    per_split = []
    for folder in folders:
        set_seed(args.seed)
        print(f"\n=== {args.model} | {args.bbdd} | {folder} ===")
        (_, _, X_test_t, _, _, X_test_s, _, _, y_test) = utils.load_data(
            device, folder, T, params["SG"], args.bbdd, norm, norm_static)
        S_temporal, S_static = load_graphs(args.model, args.bbdd, params["way_to_build_graph"],
                                           folder, norm, T, params["path_S_static"],
                                           X_test_s.shape[1], device)
        model = build_model(args.model, hyper_all[folder], S_temporal, S_static,
                            seeds[folder], bool(params["use_static"]), device)
        model.load_state_dict(torch.load(weights_path(args.model, exp_dir, args.bbdd, folder),
                                         map_location=device))
        model.eval()

        masks = batch_explain(model, X_test_t, X_test_s, y_test, explainer_params,
                              {"SG": params["SG"]}, device)
        out_dir = os.path.join(args.output_dir, args.bbdd, args.model, folder)
        os.makedirs(out_dir, exist_ok=True)
        for name, m in zip(["mean_temporal_pos", "mean_temporal_neg",
                            "mean_static_pos", "mean_static_neg"], masks):
            np.save(os.path.join(out_dir, f"{name}.npy"), m)
        per_split.append(masks)

    out_dir = os.path.join(args.output_dir, args.bbdd, args.model, "all")
    os.makedirs(out_dir, exist_ok=True)
    for k, name in enumerate(["mean_temporal_pos", "mean_temporal_neg",
                              "mean_static_pos", "mean_static_neg"]):
        np.save(os.path.join(out_dir, f"{name}.npy"),
                np.mean([m[k] for m in per_split], axis=0))
    print(f"\nMasks saved under {os.path.join(args.output_dir, args.bbdd, args.model)}")


if __name__ == "__main__":
    main()
