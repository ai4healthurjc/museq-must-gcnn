import torch
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
    device = torch.device(f"cuda:{params['device_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    way_to_build_graph = params["way_to_build_graph"]
    SG = params["SG"]

    best_params_file = f"{HERE}/hyperparameters/{bbdd}/{way_to_build_graph}/App1_temporal.json"
    tester = utils.Tester(best_params_file)

    # Sobrescribir dataset_name con bbdd
    params["dataset_name"] = bbdd

    tester.train_and_save(params, device, train_epochs=params["n_epochs"])
    results, _ = tester.test(params, device)

    keys = list(results.keys())
    results_by_split = {}

    for idx, folder in enumerate(params["folders"]):
        print(f"\n===== SPLIT {folder} =====")
        split_dict = {}
        for key in keys:
            value = results[key][idx]
            split_dict[key] = value
            print(f"{key}: {value * 100:.2f}%")
        results_by_split[folder] = split_dict

    print("\n=== Overall averages ===")
    summary = []
    for key in keys:
        avg = np.mean(results[key]) * 100
        std = np.std(results[key]) * 100
        print(f"{key}: {avg:.2f} ± {std:.2f}")
        summary.append(f"{avg:.2f}±{std:.2f}")
    print("\nLaTeX format:", " & ".join(summary), "\\\\")

    output_dir = f"{HERE}/results/{bbdd}/temporal"
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, "results.json")
    with open(output_json_path, 'w') as out_file:
        json.dump(results_by_split, out_file, indent=4)
    print(f"\nSaved: {output_json_path}")


if __name__ == "__main__":
    main()
