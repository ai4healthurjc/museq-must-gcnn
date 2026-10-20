import pandas as pd
import numpy as np
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, text
from scipy.sparse import kronsum
import warnings
warnings.filterwarnings("ignore")

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
debug_plot_figures = _args.debug

# ── Funciones ─────────────────────────────────────────────────────────────────
def exp_kernel(train, sigma):
    """
    Computes the exponential kernel matrix for the given data and sigma.

    Parameters:
    train (np.array or pd.DataFrame): The input data matrix.
    sigma (float): The kernel bandwidth parameter.

    Returns:
    pd.DataFrame: The computed kernel matrix, rounded to 6 decimal places.
    """
    matrix_train = np.exp(-(train**2) / (2 * (sigma**2)))
    x = pd.DataFrame(matrix_train)
    x = np.round(x, 6)
    eigenvalues = np.linalg.eig(x)
    matrix = np.array(x)
    if not (np.sum(np.abs(eigenvalues[0]) > 0) == x.shape[0]) and (np.array_equal(matrix, matrix.T)):
        print("==============================")
        print("DOES NOT satisfy kernel condition")
        print("==============================")
    return x


def buildAndExportGraph(shiftOperator, params):
    """
    Builds a weighted undirected graph from the shift operator matrix and node labels.

    Parameters:
    shiftOperator (np.array or pd.DataFrame): The shift operator matrix representing the graph.
    params (list of str): List of node labels corresponding to the matrix indices.

    Returns:
    networkx.Graph: The constructed weighted undirected graph.
    """
    aux = pd.DataFrame(data=shiftOperator, columns=params)
    aux.index = params
    G = nx.Graph()
    for i in range(aux.shape[0]):
        for j in range(aux.shape[0]):
            G.add_edges_from([(params[i], params[j])], weight=aux.iloc[i, j])
    return G


# ── Parámetros ────────────────────────────────────────────────────────────────
norm               = NORM
buildGraph         = "hgd"
folders            = [f"s{i}" for i in range(1, N_SPLITS + 1)]

# keys/binary/F leidos del PROPIO estatico (no hardcodear)
_static0 = pd.read_csv(f"../../../DATA/{bbdd}/s1/X_train_static_{norm}.csv")
keys               = list(_static0.columns)
binary             = [c for c in keys if set(pd.unique(_static0[c].dropna())) <= {0, 1}]
continues          = [v for v in keys if v not in binary]

threshold_val_init = 0.975
th_folder          = "th_0975"
save_results       = True

numberOfTimeSteps  = 1
numberOfFeatures   = len(keys)
T = numberOfTimeSteps
F = numberOfFeatures
print(f"  keys={len(keys)} binary={len(binary)} F={F} (norm={norm})")

# ── Crear carpetas de salida ──────────────────────────────────────────────────
for folder in folders:
    os.makedirs(f"./{bbdd}/hgd/{folder}", exist_ok=True)

# ── Representación del grafo estático ─────────────────────────────────────────
dicc_thresholds_eachTS = {}
values_to_smoothness_bysplit = []

for c in range(len(folders)):
    print("====> Folder:" + str(folders[c]) + " <====")

    matrix = pd.DataFrame(np.zeros((F * T, F * T), dtype=float))
    thresholds_by_timeStep = []
    values_to_smoothness = []

    nt = 0
    data = pd.read_csv(f"../step1_graphEstimation/estimatedGraphs/{bbdd}/{buildGraph}/{folders[c]}/adj_X_train_static_{norm}.csv")
    data = data.fillna(0)

    K = exp_kernel(data, 5.5)
    K = K - np.eye(K.shape[0])

    G = nx.from_numpy_array(np.array(K), create_using=nx.DiGraph)
    fixed_pos = nx.spring_layout(G, seed=42, k=1.5, scale=2.5)

    for nt in range(numberOfTimeSteps):
        print("========> Time Step: " + str(nt))
        data = pd.read_csv(f"../step1_graphEstimation/estimatedGraphs/{bbdd}/{buildGraph}/{folders[c]}/adj_X_train_static_{norm}.csv")
        data = data.fillna(0)

        K = exp_kernel(data, 5.5)
        K = K - np.eye(K.shape[0])          # zero diagonal (no self-loops)

        print("Threshold value:", threshold_val_init)
        print("Number of non-zero values before applying the threshold:", np.count_nonzero(K))
        s = K.copy()
        min_value = s.min().min()
        max_value = s.max().max()
        s = (s - min_value) / (max_value - min_value)

        print(s)

        ## Very small graph: remove thresholding
        ##s[np.abs(s) < threshold_val_init] = 0

        edges_aft = np.count_nonzero(s)
        values_to_smoothness.append(edges_aft)
        print("Number of non-zero values after applying the threshold:", edges_aft)
        print("%:", (edges_aft * 100) / (numberOfFeatures * numberOfFeatures))
        thresholds_by_timeStep.append((edges_aft * 100) / (numberOfFeatures * numberOfFeatures))

        if debug_plot_figures:
            plt.figure()
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            img = ax.imshow(s, extent=[0, numberOfFeatures, numberOfFeatures, 0])
            ax.set_xticks(range(0, numberOfFeatures+1, 5))
            ax.set_yticks(range(0, numberOfFeatures+1, 5))
            ax.set_xticklabels(range(0, numberOfFeatures+1, 5), rotation=90, fontsize=23)
            ax.set_yticklabels(range(0, numberOfFeatures+1, 5), fontsize=23)
            cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=23)
            plt.tight_layout(pad=0)
            plt.show()

            G = nx.from_numpy_array(np.array(s), create_using=nx.DiGraph)
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) if len(G.edges()) > 0 else ([], [])
            options = {
                "edgelist": edges,
                "edge_color": weights,
                "width": np.array(weights) * 1 if weights else [],
                "alpha": 1,
            }
            node_colors = ['lightblue' if keys[node] in binary else 'lightgreen' for node in G.nodes()]
            node_sizes = [700 if G.degree(node) > 0 else 500 for node in G.nodes()]

            fig2, ax2 = plt.subplots(figsize=(16, 8))
            nx.draw_networkx_nodes(G, fixed_pos, node_color=node_colors, node_size=node_sizes, ax=ax2)
            if edges:
                nx.draw_networkx_edges(G, fixed_pos, **options, ax=ax2)
            labels = {node: node for node in G.nodes()}
            nx.draw_networkx_labels(G, fixed_pos, labels=labels, font_size=14, font_weight='bold', ax=ax2)
            plt.tight_layout()
            plt.show()

        matrix.iloc[nt * F: (nt + 1) * F, nt * F: (nt + 1) * F] = s.values

    dicc_thresholds_eachTS[folders[c]] = thresholds_by_timeStep
    values_to_smoothness_bysplit.append(values_to_smoothness)

    for i in range(F, F * T):
        matrix.iloc[i, i - F] = 1

    if debug_plot_figures:
        plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        img = ax.imshow(matrix)
        fig.colorbar(img)
        plt.tight_layout()
        plt.show()

    if save_results:
        out_path = f"./{bbdd}/hgd/{folders[c]}/adj_X_train_static_{norm}_th_{threshold_val_init}.csv"
        pd.DataFrame(matrix).to_csv(out_path, index=False)
        print(f"  → Guardado en {out_path}")

print(f"✓ Representación de grafos estáticos completada para {bbdd}.")