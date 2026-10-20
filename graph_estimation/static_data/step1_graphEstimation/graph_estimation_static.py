import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import hgd

# ── Argumentos: --bbdd <nombre> [--norm <normalizacion>] [--n_splits 10] [--debug] ──
import argparse as _argparse
_DEFAULT_NORM = {"AMR": "minMax_per_patient", "physionet2012": "robustNorm"}
_ap = _argparse.ArgumentParser()
_ap.add_argument("bbdd_pos", nargs="?", default=None)
_ap.add_argument("--bbdd", default=None)
_ap.add_argument("--norm", default=None)
_ap.add_argument("--n_splits", type=int, default=10)
_ap.add_argument("--debug", action="store_true")
_args = _ap.parse_args()
bbdd = _args.bbdd or _args.bbdd_pos
if bbdd is None:
    _ap.error("indica la BBDD: --bbdd AMR  o  --bbdd physionet2012")
NORM = _args.norm or _DEFAULT_NORM.get(bbdd, "minMax_per_patient")
N_SPLITS = _args.n_splits
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # rutas relativas a este script
print(f"✓ BBDD seleccionada: {bbdd}  (norm={NORM}, splits={N_SPLITS})")

# ── Parámetros ────────────────────────────────────────────────────────────────
folders          = [f"s{i}" for i in range(1, N_SPLITS + 1)]
# Estatico sobre one-hot (robustNorm). keys/binary se leen del PROPIO CSV.
norm             = NORM
numberOfTimeStep = 1

_static0 = pd.read_csv(f"../../../DATA/{bbdd}/s1/X_train_static_{norm}.csv")
keys     = list(_static0.columns)
# binaria = columna con solo valores {0,1}
binary   = [c for c in keys if set(pd.unique(_static0[c].dropna())) <= {0, 1}]
continuous = [v for v in keys if v not in binary]
print(f"  keys={len(keys)} binary={len(binary)} (norm={norm})")

# ── Crear carpetas de salida ──────────────────────────────────────────────────
for folder in folders:
    os.makedirs(f"./estimatedGraphs/{bbdd}/hgd/{folder}", exist_ok=True)

# ── Estimación del grafo estático ─────────────────────────────────────────────
for c, folder in enumerate(folders):
    print(f"====> Folder: {folder} <====")

    X_train = pd.read_csv(f"../../../DATA/{bbdd}/{folder}/X_train_static_{norm}.csv")
    X_train = X_train.to_numpy()
    X_train = X_train.reshape(X_train.shape[0], numberOfTimeStep, X_train.shape[1])

    print(f"  shape antes de reshape: {X_train.shape}")

    X_train = hgd.minmax_for_hgd(X_train)
    X_train = hgd.reshape_patients_by_features(X_train, keys, numberOfTimeStep)
    print(f"  shape después de reshape: {X_train.shape}")

    for nt in range(numberOfTimeStep):
        x = X_train[:, nt, :]

        mat_by_time_step = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                mat_by_time_step[i, j] = hgd.hgd_distance(
                    x[i], x[j], keys[i], keys[j], binary, continuous
                )

        S = hgd.diagonal_to_zero(mat_by_time_step)

        out_path = f"./estimatedGraphs/{bbdd}/hgd/{folder}/adj_X_train_static_{norm}.csv"
        pd.DataFrame(S, columns=keys, index=keys).to_csv(out_path, index=False)
        print(f"  → Guardado en {out_path}")

print(f"✓ Estimación de grafos estáticos completada para {bbdd}.")