"""Aggregate the per-split results of every model and compare them statistically.

Reads every ``results.json`` produced by the ``*_testing.py`` scripts for one
dataset, builds the per-split table and the mean +- standard deviation summary,
plots the evolution of the main metrics across splits, and runs the paired
comparison of a reference model against all the others.

Usage:
    python analyse_results.py --bbdd physionet2012
    python analyse_results.py --bbdd AMR --reference MuSeqGCNN --n_splits 10
"""
import argparse
import glob
import itertools
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from scipy.stats import t, ttest_rel, wilcoxon   # noqa: E402

plt.rcParams["figure.dpi"] = 120

# Folder name -> model name reported in the paper
FOLDER_TO_MODEL = {
    "2_GCNN_RNN": "MuSeqGCNN",
    "2_XMuST_GCNN": "MuSTGCNN",
    "gru": "gru", "lstm": "lstm", "mlp": "mlp", "lr": "lr",
    "mamba": "mamba", "transformer": "transformer",
    "mlp_gru_multimodal": "mlp_gru_multimodal", "gcnn": "gcnn",
    "temporal": "gcnn_temporal", "static": "gcnn_static",
}

SKIP_FOLDERS = {"results", "plots", "hyperparameters", "models", "codecarbon_logs"}

METRICS = ["test_acc", "roc_auc", "aucpr", "f1_score", "sensitivity",
           "specificity", "training_time", "inference_time"]

PLOT_METRICS = ["roc_auc", "aucpr", "sensitivity", "specificity"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbdd", required=True, help="dataset name, e.g. physionet2012")
    p.add_argument("--results_dir", default=None,
                   help="root to search for results.json (default: ../experiments)")
    p.add_argument("--n_splits", type=int, default=10,
                   help="number of splits to keep, s1..sN (default: 10)")
    p.add_argument("--reference", default="MuSeqGCNN",
                   help="model used as reference in the paired comparison")
    p.add_argument("--metric", default="roc_auc",
                   help="metric used in the paired comparison (default: roc_auc)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="significance level (default: 0.05)")
    return p.parse_args()


def model_name_from_path(filepath, bbdd):
    parts = filepath.split(os.sep)
    for part in reversed(parts[:-1]):
        if part in FOLDER_TO_MODEL:
            return FOLDER_TO_MODEL[part]
    skip = SKIP_FOLDERS | {bbdd}
    return next((p for p in reversed(parts[:-1]) if p not in skip), parts[-2])


def split_number(labels):
    """Map a split label to its number, aligned with the input index.

    ``s7`` -> 7, ``gnn_rnn_1_s7`` -> 7, ``7`` -> 7. Labels with no number
    become NaN; the returned Series keeps the original index so that it can be
    used directly as a boolean mask.
    """
    s = pd.Series(labels).astype(str)
    num = s.str.extract(r"s(\d+)$", expand=False)
    num = num.fillna(s.str.extract(r"(\d+)$", expand=False))
    return pd.to_numeric(num, errors="coerce")


def load_results(results_dir, bbdd, n_splits):
    pattern = os.path.join(results_dir, "**", bbdd, "**", "results.json")
    frames = []
    for filepath in sorted(glob.glob(pattern, recursive=True)):
        model = model_name_from_path(filepath, bbdd)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        df = pd.DataFrame(data).T
        df.index.name = "split_label"
        df = df.reset_index()
        df["model"] = model
        df["split"] = split_number(df["split_label"])
        frames.append(df)
        print(f"  loaded {len(df):3d} splits from {os.path.relpath(filepath)} -> {model}")

    if not frames:
        raise SystemExit(f"No results found for '{bbdd}' under {results_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["split"])
    df["split"] = df["split"].astype(int)
    df = df[df["split"] <= n_splits]

    for m in [c for c in METRICS if c in df.columns]:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    counts = df.groupby("model")["split"].nunique()
    incomplete = counts[counts != n_splits]
    if len(incomplete):
        print("\n  WARNING: these models do not have exactly "
              f"{n_splits} splits and are excluded from the paired comparison:")
        for model, n in incomplete.items():
            print(f"    {model}: {n} splits")
    return df


# --------------------------------------------------------------------------- #
# Summary and plots
# --------------------------------------------------------------------------- #
def summarise(df, metrics):
    mean = df.groupby("model")[metrics].mean()
    std = df.groupby("model")[metrics].std()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = (mean[m].round(4).astype(str) + " ± " + std[m].round(4).astype(str))
    return out.sort_index()


def plot_evolution(df, metric, output_dir):
    models = sorted(df["model"].unique())
    cols = 2
    rows = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), squeeze=False)
    for ax, model in itertools.zip_longest(axes.ravel(), models):
        if model is None:
            ax.axis("off")
            continue
        sub = df[df["model"] == model].sort_values("split")
        ax.plot(sub["split"], sub[metric], marker="o", lw=1.2)
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("split")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{metric} across splits", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"evolution_{metric}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Paired statistical comparison
# --------------------------------------------------------------------------- #
def _signed_rank_k(n, alpha=0.05):
    """Index of the exact Wilcoxon confidence interval, by enumeration."""
    ranks = np.arange(1, n + 1)
    dist = np.array([np.dot(s, ranks)
                     for s in itertools.product([0, 1], repeat=n)])
    total = int(ranks.sum())
    cdf = np.array([(dist <= c).mean() for c in range(total + 1)])
    below = np.where(cdf <= alpha / 2)[0]
    return int(below.max()) + 1 if below.size else 1


def hodges_lehmann(d, alpha=0.05):
    """Hodges-Lehmann estimate of the median difference and its exact CI.

    The estimate is the median of the n(n+1)/2 Walsh averages of the paired
    differences; the interval is the distribution-free one induced by the
    signed-rank statistic, so it is the interval that matches the Wilcoxon
    test used for the p-values.
    """
    n = d.size
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    k = _signed_rank_k(n, alpha)
    return float(np.median(walsh)), float(walsh[k - 1]), float(walsh[-k])


def mean_difference(d, alpha=0.05):
    """Mean paired difference and its 95% confidence interval.

    The interval is the usual Student-t interval of the paired differences,
    ``dbar +- t_{1-alpha/2, n-1} * sd(d) / sqrt(n)``. The point estimate is the
    difference of the two per-split means, i.e. exactly the difference between
    the two entries of the results table, which is what is reported in the
    paper.
    """
    n = d.size
    dbar = float(d.mean())
    if n < 2:
        return dbar, dbar, dbar
    se = float(d.std(ddof=1) / np.sqrt(n))
    half = float(t.ppf(1 - alpha / 2, n - 1) * se)
    return dbar, dbar - half, dbar + half


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation."""
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_comparison(df, reference, metric, n_splits, alpha=0.05):
    wide = df.pivot_table(index="split", columns="model", values=metric)
    wide = wide.dropna(axis=1)                       # keep complete models only
    if reference not in wide.columns:
        raise SystemExit(f"Reference model '{reference}' has no complete results "
                         f"for '{metric}'. Available: {list(wide.columns)}")
    if wide.shape[0] != n_splits:
        print(f"  note: {wide.shape[0]} splits are common to all models")

    ref = wide[reference].to_numpy(dtype=float) * 100
    rows = []
    for model in wide.columns:
        if model == reference:
            continue
        other = wide[model].to_numpy(dtype=float) * 100
        d = ref - other
        if np.allclose(d, 0):
            rows.append((model, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0))
            continue
        delta, lo, hi = mean_difference(d, alpha)
        hl, hl_lo, hl_hi = hodges_lehmann(d, alpha)
        _, p_w = wilcoxon(ref, other, alternative="two-sided", method="exact")
        _, p_t = ttest_rel(ref, other)
        rows.append((model, delta, lo, hi, hl, hl_lo, hl_hi,
                     rank_biserial(d), float(p_w), float(p_t)))

    out = pd.DataFrame(rows, columns=["model", "delta_mean",
                                      "ci_low", "ci_high",
                                      "delta_hl", "hl_low", "hl_high",
                                      "rank_biserial",
                                      "p_wilcoxon", "p_ttest"])
    out["p_holm"] = holm(out["p_wilcoxon"].to_numpy())
    out["significant"] = out["p_holm"] < alpha
    return out.sort_values("delta_mean").reset_index(drop=True), wide


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(here, "..", "experiments")
    output_dir = os.path.join(here, args.bbdd)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Loading results for '{args.bbdd}' ===")
    df = load_results(results_dir, args.bbdd, args.n_splits)
    metrics = [m for m in METRICS if m in df.columns]

    # 1. per-split table (the raw material of everything else)
    per_split = df[["model", "split"] + metrics].sort_values(["model", "split"])
    per_split.to_csv(os.path.join(output_dir, "per_split_results.csv"), index=False)
    print(f"\n  saved {os.path.join(output_dir, 'per_split_results.csv')}")

    # 2. summary
    print(f"\n=== Summary over splits 1-{args.n_splits} (mean ± std) ===")
    summary = summarise(df, metrics)
    print(summary.to_string())
    summary.to_csv(os.path.join(output_dir, "summary.csv"))

    # 3. evolution plots
    print("\n=== Evolution plots ===")
    for metric in [m for m in PLOT_METRICS if m in df.columns]:
        plot_evolution(df, metric, os.path.join(output_dir, "plots"))

    # 4. paired comparison
    print(f"\n=== Paired comparison: {args.reference} vs the rest "
          f"({args.metric}, exact Wilcoxon signed-rank, Holm-corrected) ===")
    stats, wide = paired_comparison(df, args.reference, args.metric,
                                    args.n_splits, args.alpha)
    with pd.option_context("display.float_format", lambda v: f"{v:8.4f}"):
        print(stats.to_string(index=False))
    stats.to_csv(os.path.join(output_dir, f"paired_comparison_{args.metric}.csv"),
                 index=False)
    print("\n  delta_mean = mean paired difference in points, with the paired-t "
          "interval [ci_low, ci_high];")
    print("  delta_hl   = Hodges-Lehmann median difference, with the exact "
          "signed-rank interval [hl_low, hl_high].")

    n = wide.shape[0]
    floor = 2 / (2 ** n) * (len(stats))
    print(f"\n  n = {n} paired splits. The exact test cannot return a two-sided "
          f"p-value below {2 / (2 ** n):.4f},")
    print(f"  hence no Holm-corrected value below {min(floor, 1.0):.4f} across "
          f"{len(stats)} comparisons.")
    disagree = stats[(stats["p_wilcoxon"] < args.alpha)
                     != (stats["p_ttest"] < args.alpha)]
    if len(disagree):
        print("\n  WARNING: Wilcoxon and paired t-test disagree for:",
              ", ".join(disagree["model"]))
    else:
        print("  Wilcoxon and paired t-test agree on every comparison.")

    print(f"\nDone. Output written to {output_dir}")


if __name__ == "__main__":
    main()
