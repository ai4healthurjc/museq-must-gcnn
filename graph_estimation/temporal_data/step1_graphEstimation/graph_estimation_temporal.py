import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed

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
folders = [f"s{i}" for i in range(1, N_SPLITS + 1)]

norm = NORM

# ── keys / binary / T leidos del PROPIO dataset (no hardcodear) ───────────────
# Se leen del primer split: keys.csv y variable_types.csv los genera el
# generador de particiones; numberOfTimeStep se deduce de X.shape[1].
_meta_dir = f"../../../DATA/{bbdd}/s1"
keys = pd.read_csv(f"{_meta_dir}/keys.csv")["keys"].tolist()

try:
    _vt = pd.read_csv(f"{_meta_dir}/variable_types.csv")
    binary = _vt.loc[_vt["type"] == "binary", "variable"].tolist()
except FileNotFoundError:
    binary = []

continuous = [v for v in keys if v not in binary]

_X0 = np.load(f"{_meta_dir}/X_train_tensor_{norm}.npy")
numberOfTimeStep = _X0.shape[1]
assert _X0.shape[2] == len(keys), (
    f"keys ({len(keys)}) no casa con F del tensor ({_X0.shape[2]})"
)
print(f"  keys={len(keys)} binary={len(binary)} T={numberOfTimeStep}")

# ── Función para paralelizar filas ───────────────────────────────────────────
def compute_row(i, x, keys, binary, continuous):
    row = []
    for j in range(x.shape[0]):
        row.append(hgd.hgd_distance(
            x[i], x[j], keys[i], keys[j], binary, continuous
        ))
    return row

# ── Crear carpetas de salida ──────────────────────────────────────────────────
for folder in folders:
    os.makedirs(f"./estimatedGraphs/{bbdd}/hgd/{folder}", exist_ok=True)

# ── Estimación del grafo temporal ─────────────────────────────────────────────
for c, folder in enumerate(folders):
    print(f"====> Folder: {folder} <====")

    X_train = np.load(f"../../../DATA/{bbdd}/{folder}/X_train_tensor_{norm}.npy")
    X_train = hgd.minmax_for_hgd(X_train)
    X_train = hgd.reshape_patients_by_features(X_train, keys, numberOfTimeStep)
    print(f"  X_train shape después de reshape: {X_train.shape}")

    for nt in range(numberOfTimeStep):

        # ── Opción 3: saltar si ya existe ────────────────────────────────────
        out_path = f"./estimatedGraphs/{bbdd}/hgd/{folder}/adj_X_train_TS_{nt}_{norm}.csv"
        if os.path.exists(out_path):
            print(f"  → TS {nt} ya existe, saltando")
            continue

        x = X_train[:, nt, :]

        # ── Opción 1: paralelizar filas con joblib ───────────────────────────
        rows = Parallel(n_jobs=-1)(
            delayed(compute_row)(i, x, keys, binary, continuous)
            for i in range(x.shape[0])
        )
        mat_by_time_step = np.array(rows)

        S = hgd.diagonal_to_zero(mat_by_time_step)
        S = np.nan_to_num(S, nan=0.0)  # NaN->0 (labs constantes con minMax_per_patient meten NaN en la HGD)
        pd.DataFrame(S, columns=keys, index=keys).to_csv(out_path, index=False)
        print(f"  → TS {nt} guardado en {out_path}")

print(f"✓ Estimación de grafos temporales completada para {bbdd}.")
