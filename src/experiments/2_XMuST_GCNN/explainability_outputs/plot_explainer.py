import numpy as np
import matplotlib.pyplot as plt
from typing import List
from mpl_toolkits.axes_grid1 import make_axes_locatable


VARIABLE_NAMES: List[str] = [
    "AFA", "AFE", "AFO", "AFS", "AMG", "CAR", "CF1", "CF2", "CF3", "CFO", "CPS", "GCC", 
    "GLI", "IBL", "LIN", "LIP", "MAC", "MON", "NTI", "OTR", "OXA", "Otros", "PEN", "POL", 
    "QUI", "SUL", "TMS", "TTC", "MV_hours", "ICU_hours", "n_patients", "n_patientsAMR",
    "CAR$_{n}$", "IBL$_{n}$", "AFO$_{n}$", "OXA$_{n}$", "QUI$_{n}$", "PEN$_{n}$",
    "AFA$_{n}$", "CF3$_{n}$", "GLI$_{n}$", "CPS$_{n}$", "TMS$_{n}$", "LIN$_{n}$",
    "NTI$_{n}$", "MAC$_{n}$", "OTR$_{n}$", "AMG$_{n}$", "AFE$_{n}$", "POL$_{n}$",
    "CF1$_{n}$", "GCC$_{n}$", "LIP$_{n}$", "AFS$_{n}$", "MON$_{n}$", "CFO$_{n}$",
    "Others$_{n}$", "TTC$_{n}$", "CF2$_{n}$", "SUL$_{n}$", "postural_change",
    "insuline", "artificial_nutr.", "sedation", "relaxation", "n_transf", "vasoactive_drugs", 
    "NEMS", "Tracheo_hours", "Ulcer_hours", "Hemo_hours",
    "C01 PICC 1", "C01 VC PICC 2", "C02 CVC - RY",
    "C02 CVC - RS", "C02 CVC - LS", "C02 CVC - RF", "C02 CVC - LY",
    "C02 CVC - LF", "n_catheters", "acinet$_{pc}$", "enteroc$_{pc}$", "pseud$_{pc}$",
    "staph$_{pc}$", "others$_{pc}$"
]


def load_and_plot_temporal_mask(
    mask_path: str,
    var_names: List[str] = VARIABLE_NAMES,
    figsize: tuple = (11, 40),
    cmap: str = 'coolwarm',
    output_path: str = None
):
    """
    Load a temporal mask from a .npy file and plot it as a heatmap.
    Time on the X-axis, variables on the Y-axis.
    Normalizes data to [0,1] and makes the colorbar span full height of the plot.
    """
    # Load the 1D mask
    mask = np.load(mask_path)
    n_vars = len(var_names)
    time_steps = mask.size // n_vars

    # Reshape (time_steps × n_vars) and transpose → (n_vars × time_steps)
    plot_data = mask.reshape((time_steps, n_vars)).T

    # Normalize to [0,1]
    pmin, pmax = plot_data.min(), plot_data.max()
    plot_data = (plot_data - pmin) / (pmax - pmin)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_data, aspect='auto', cmap=cmap)

    # Colorbar full height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_tick_params(labelsize=30)

    # X ticks
    if time_steps <= 50:
        xt = np.arange(time_steps)
        ax.set_xticks(xt)
        ax.set_xticklabels(xt+1, fontsize=26, rotation=90)
    else:
        step = max(1, time_steps // 50)
        xt = np.arange(0, time_steps, step)
        ax.set_xticks(xt)
        ax.set_xticklabels(xt+1, fontsize=26, rotation=90)

    # Y ticks
    if n_vars <= 50:
        yt = np.arange(n_vars)
        ax.set_yticks(yt)
        ax.set_yticklabels(var_names, fontsize=30)
    else:
        step = max(1, n_vars // 50)
        yt = np.arange(0, n_vars, step)
        ax.set_yticks(yt)
        ax.set_yticklabels([var_names[i] for i in yt], fontsize=30)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_static_masks_grouped(
    static_vars: List[str],
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    normalize: bool = True,
    figsize: tuple = (6,4),
    fontsize: int = 23,
    output_path: str = None
):
    # Ensure masks are 1D arrays
    pos_mask = np.asarray(pos_mask).flatten()
    neg_mask = np.asarray(neg_mask).flatten()
    
    # Verify lengths match
    assert len(static_vars) == len(pos_mask) == len(neg_mask), \
        "Length of static_vars must match length of masks"
    
    # Prepare data
    if normalize:
        gm = max(pos_mask.max(), neg_mask.max())
        pos_vals = (pos_mask / gm).tolist() if gm != 0 else pos_mask.tolist()
        neg_vals = (neg_mask / gm).tolist() if gm != 0 else neg_mask.tolist()
        ylabel = "Normalized importance (0–1)"
    else:
        pos_vals = pos_mask.tolist()
        neg_vals = neg_mask.tolist()
        ylabel = "Raw importance"

    N = len(static_vars)
    x = np.arange(N)  # Use numpy array for positions
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert all values to plain Python floats to avoid array issues
    pos_vals = [float(v) for v in pos_vals]
    neg_vals = [float(v) for v in neg_vals]
    
    b1 = ax.bar(x=x, height=pos_vals, width=width, label='Positive mask')
    b2 = ax.bar(x=x + width, height=neg_vals, width=width, label='Negative mask')

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(static_vars, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    # Annotate raw values
    for bars, raw in ((b1, pos_mask), (b2, neg_mask)):
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in raw], fontsize=fontsize-4)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_grouped_separate_normalization(
    static_vars: List[str],
    amr_mask: np.ndarray,
    noamr_mask: np.ndarray,
    figsize=(5, 2.5),
    fontsize=12,
    output_path: str = None
):
    # Ensure inputs are 1D numpy arrays of floats
    amr_mask   = np.asarray(amr_mask,   dtype=float).flatten()
    noamr_mask = np.asarray(noamr_mask, dtype=float).flatten()
    
    # Handle zero-division cases
    amr_max   = amr_mask.max()   if amr_mask.max()   != 0 else 1
    noamr_max = noamr_mask.max() if noamr_mask.max() != 0 else 1
    
    amr_norm   = amr_mask   / amr_max
    noamr_norm = noamr_mask / noamr_max
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(static_vars))
    width = 0.25
    
    # AMR: solid blue
    bars_amr = ax.bar(
        x - width/2,
        amr_norm,
        width,
        label="AMR",
        color="blue",
        edgecolor="black"
    )
    
    # noAMR: green fill + thin white diagonal hatching
    bars_noamr = ax.bar(
        x + width/2,
        noamr_norm,
        width,
        label="non-AMR",
        facecolor="green",
        edgecolor="black",
        linewidth=1,
        hatch='///'      # diagonal stripes
    )
    # if your matplotlib supports hatch_color, enforce white stripes:
    for bar in bars_noamr:
        try:
            bar.set_hatch_color('white')
        except AttributeError:
            pass  # older mpl will just draw hatch in edgecolor
    
    # X ticks & labels
    ax.set_xticks(x)
    ax.set_xticklabels(static_vars, rotation=0, ha="right", fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    
    # Legend & axis label
    # ax.legend(fontsize=fontsize)
    
    # Annotate raw (pre-normalization) values on top
    # for bars, raw in ((bars_amr, amr_mask), (bars_noamr, noamr_mask)):
    #     ax.bar_label(bars, labels=[f"{v:.2f}" for v in raw], fontsize=fontsize-2)
    
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_static_masks_dualaxis(
    static_vars: List[str],
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    figsize: tuple = (5,3),
    fontsize: int = 23,
    output_path: str = None
):
    # Prepare data
    N = len(static_vars)
    x = list(range(N))
    width = 0.25

    pos_vals = pos_mask.tolist()
    neg_vals = neg_mask.tolist()

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    b1 = ax1.bar(x=x, height=pos_vals, width=width, label='Positive mask')
    b2 = ax2.bar(x=[xi + width for xi in x],
                 height=neg_vals, width=width, label='Negative mask')

    ax1.set_ylabel("Positive mask importance", fontsize=fontsize)
    ax2.set_ylabel("Negative mask importance", fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)

    ax1.set_xticks(x)
    ax1.set_xticklabels(static_vars, fontsize=fontsize)

    # Combined legend
    handles = list(b1) + list(b2)
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper left', fontsize=fontsize)

    # Annotate
    for bars, raw, ax in ((b1, pos_mask, ax1), (b2, neg_mask, ax2)):
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in raw], fontsize=fontsize-4)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)

if __name__ == "__main__":
    # === Temporal heatmap ===
    fig_t, ax_t = load_and_plot_temporal_mask(
        mask_path="explainability_outputs/npy/mean_temporal_neg.npy",
        output_path="explainability_outputs/visualizations/mean_temporal_neg_normalized.pdf"
    )

    # === Static masks ===
    # replace these with your real arrays
    static_vars = ["age", "SAPS-III", "gender"]
    pos_mask = np.load('explainability_outputs/npy/mean_static_pos.npy')
    neg_mask = np.load('explainability_outputs/npy/mean_static_neg.npy')

    # Option A: normalized to [0,1]
    fig_s, ax_s = plot_grouped_separate_normalization(
        static_vars, pos_mask, neg_mask,
        output_path="explainability_outputs/visualizations/static_masks_grouped_norm.pdf"
    )

    plt.show()