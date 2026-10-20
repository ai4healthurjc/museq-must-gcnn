# Interpretable Multimodal Spatio-Temporal Graph Networks for Irregular Clinical Time Series

Reference implementation of **MuSeq-GCNN** and **MuST-GCNN**, two multimodal
spatio-temporal graph architectures that jointly model irregularly sampled
physiological time series and time-invariant patient attributes, together with
the adapted GNNExplainer used to interpret them.

This repository contains the complete pipeline reported in the paper:
graph estimation, graph representation, hyper-parameter optimisation, training,
evaluation and explainability, for both proposed models and for all baselines.

![Python](https://img.shields.io/badge/python-3.8%E2%80%933.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A51.13-ee4c2c)
![Licence](https://img.shields.io/badge/licence-MIT-green)

---

## Method

Both architectures take two inputs per patient: a temporal tensor of `F`
irregularly sampled physiological variables observed over `T` time steps, and a
vector of `D` time-invariant attributes. Variables are treated as the nodes of a
graph whose edges are estimated from the data, so the models exploit the
relations between clinical variables rather than assuming independence.

**Graphs.** For every pair of variables, the Heterogeneous Gower Distance (HGD)
is computed on the training patients (separately at each time step for the
time-varying variables) and converted into a similarity with an exponential
kernel. Temporal graphs are pruned below a threshold; the static graph, which has
only `D` nodes, is not pruned. The per-time-step graphs `A_t` are assembled into a
unified spatio-temporal graph `A_ST` whose diagonal blocks are the `A_t` and
whose blocks immediately below the diagonal are identities, so each node
`(f, t)` aggregates information from `(f, t-1)`. Adjacency matrices are
normalised as `Â = D̃^{-1/2}(A + I)D̃^{-1/2}`.

**Two ways of modelling time.** The models differ only in how the temporal
branch is organised:

| Model | Temporal branch | Graph size |
|---|---|---|
| **MuSeq-GCNN** (successive) | one GCNN per time step over `A_t`; node embeddings concatenated (`F·U`) and fed to a GRU, whose states are averaged | `F` nodes, `T` graphs |
| **MuST-GCNN** (simultaneous) | a single GCNN over `A_ST` with one scalar output per node, concatenated into `z ∈ R^{F·T}` | `F x T` nodes, 1 graph |

**Fusion.** A gated attention mechanism combines the two modalities per patient,
`alpha_p = sigmoid(w^T [static ; temporal] + b)` and
`e_p = alpha_p * static + (1 - alpha_p) * temporal`, so `alpha_p` is the weight
placed on the static branch. It is written to `xai_intrinsic.json` at test time
and is the quantity analysed in the paper.

**Interpretability.** An adapted GNNExplainer learns, for each test patient, two
separate masks — one of shape `F x T` over the temporal input and one of shape
`D` over the static attributes — with the trained model frozen. The masks are
fitted to the label predicted by the model with size (L1) and entropy
regularisation, and are then averaged within each outcome group.

---

## Contents

```text
.
├── DATA/                         # empty by design — see DATA/README.md
├── graph_estimation/             # step 1: estimate graphs, step 2: build representations
│   ├── static_data/
│   │   ├── step1_graphEstimation/   hgd.py, graph_estimation_static.py
│   │   └── step2_graphRepresentation/ graph_representation_static.py
│   └── temporal_data/
│       ├── step1_graphEstimation/   hgd.py, graph_estimation_temporal.py
│       └── step2_graphRepresentation/ graph_representation_temporal.py
├── src/
│   ├── utils/architectures/      # model definitions and training/evaluation utilities
│   │   ├── MuSeqGCNN/            # successive architecture (proposed)
│   │   ├── MuSTGCNN/             # simultaneous architecture (proposed)
│   │   ├── gcnn/ gru/ lstm/ lr/ mlp/ mamba/ transformer/ mlp_gru_multimodal/
│   ├── experiments/              # entry points, one folder per model family
│   │   ├── 2_GCNN_RNN/           # MuSeq-GCNN
│   │   ├── 2_XMuST_GCNN/         # MuST-GCNN
│   │   └── baselines/
│   ├── explainability/explain.py # adapted GNNExplainer (both models, all splits)
│   └── results_analysis/         # aggregation and paired statistical comparison
└── tests/
    └── test_smoke.py             # builds both models on synthetic data
```

Each model family has two entry points, `*_optimize.py` (hyper-parameter search
with Optuna) and `*_testing.py` (training with the selected hyper-parameters and
evaluation on the test set), plus a `load_config*.json` describing the datasets.

---

## Installation

```bash
git clone https://github.com/ai4healthurjc/museq-must-gcnn.git
cd museq-must-gcnn
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.8–3.11. A CUDA-capable GPU is recommended for MuST-GCNN,
whose spatio-temporal graph has `F x T` nodes; the remaining models run on CPU.

Check that the models build correctly:

```bash
python tests/test_smoke.py
```

Two dependencies are optional and commented out in `requirements.txt`:
`codecarbon`, used only to log energy consumption during the hyper-parameter
search (the scripts skip it silently if absent), and `mamba-ssm`, required only
by the Mamba baseline.

---

## Data

The `DATA/` folder is empty by design. The AMR cohort comes from the University
Hospital of Fuenlabrada and contains confidential patient information that
cannot be released; see `DATA/README.md`. The PhysioNet/CinC Challenge 2012
dataset is public and can be obtained from
<https://physionet.org/content/challenge-2012/>.

Each dataset is expected under `DATA/<dataset>/<split>/` with one folder per
split (`s1` … `s10`) containing:

| File | Content |
|---|---|
| `X_{train,val,test}_tensor_{norm}.npy` | temporal tensor, shape `(N, T, F)` |
| `X_{train,val,test}_static_{norm}.csv` | static attributes, shape `(N, D)` |
| `y_{train,val,test}_tensor_{norm}.csv` | labels, one row per patient and time step |
| `{train,val,test}_ids.csv` | patient identifiers of each subset |
| `keys.csv`, `variable_types.csv` | channel names and variable types |

`{norm}` is the normalisation scheme declared in the configuration file
(`robustNorm` for PhysioNet, `minMax_per_patient` for AMR); every model,
proposed or baseline, uses the same scheme on a given dataset. The training subset
is balanced by random undersampling of the majority class.

The two tasks have the following shapes:

| | AMR | PhysioNet/CinC 2012 |
|---|---|---|
| Task | antimicrobial resistance during the ICU stay | in-hospital mortality |
| Temporal channels `F` | 85 | 74 (37 variables + 37 missingness masks) |
| Time steps `T` | 14 | 24 (2-hour bins over 48 h) |
| Static attributes `D` | 3 | 8 (ICU type one-hot encoded) |
| Splits | 10 | 10 |

Missingness masks are treated as ordinary channels, so the graph is estimated
over variables and masks jointly and the model can learn from the pattern of
missingness itself.

---

## Reproducing the experiments

The pipeline has four stages, and each stage consumes the output of the previous
one. All commands take the dataset name as `--bbdd`.

**1. Graph estimation.** Computes the pairwise Heterogeneous Gower Distance
between clinical variables on the training patients of each split (one matrix
per time step for the temporal variables).

```bash
python graph_estimation/static_data/step1_graphEstimation/graph_estimation_static.py --bbdd physionet2012
python graph_estimation/temporal_data/step1_graphEstimation/graph_estimation_temporal.py --bbdd physionet2012
```

**2. Graph representation.** Converts the distances into kernel similarities,
prunes the temporal graphs, saves the per-time-step graphs used by MuSeq-GCNN and
the temporal GCNN baseline, and assembles them into the unified
spatio-temporal graph used by MuST-GCNN.

```bash
python graph_estimation/static_data/step2_graphRepresentation/graph_representation_static.py --bbdd physionet2012
python graph_estimation/temporal_data/step2_graphRepresentation/graph_representation_temporal.py --bbdd physionet2012
```

**3. Hyper-parameter optimisation.** One Optuna study per split, selecting on
validation AUCROC. Writes `hyperparameters/<dataset>/`.

```bash
cd src/experiments/2_GCNN_RNN   && python MuSeqGCNN_optimize.py --bbdd physionet2012
cd src/experiments/2_XMuST_GCNN && python MuSTGCNN_optimize.py  --bbdd physionet2012
```

**4. Training and evaluation.** Trains with the selected hyper-parameters and
evaluates on the test set of each split. Writes `results/<dataset>/`, including
`results.json` with the metrics of every split and `xai_intrinsic.json` with the
per-patient modality gate.

```bash
cd src/experiments/2_GCNN_RNN   && python MuSeqGCNN_testing.py --bbdd physionet2012
cd src/experiments/2_XMuST_GCNN && python MuSTGCNN_testing.py  --bbdd physionet2012
```

Baselines follow the same two steps from their own folder, for instance
`src/experiments/baselines/gru/optimize.py` and `testing.py`.

**5. Explainability.** Applies the adapted GNNExplainer to the trained models
of every split and saves the temporal and static masks per split and averaged
over splits (`explainability_outputs/<dataset>/<model>/`).

```bash
python src/explainability/explain.py --bbdd physionet2012 --model MuSeqGCNN
python src/explainability/explain.py --bbdd physionet2012 --model MuSTGCNN
```

**6. Aggregation and statistical comparison.** Summarises the metrics of all
models across splits and runs the paired comparison of the reference model
against every other architecture (exact Wilcoxon signed-rank test with
Hodges--Lehmann effect sizes and Holm correction, plus a paired t-test as a
robustness check).

```bash
python src/results_analysis/analyse_results.py --bbdd physionet2012
```

---

## Configuration

Each experiment folder holds a `load_config*.json` with a block of parameters
shared by all datasets and a `datasets` block that overrides them per dataset.
The fields that matter most:

| Field | Meaning |
|---|---|
| `folders` | split names, `s1` … `s10` |
| `seed` | one initialisation seed per split, paired positionally with `folders` |
| `numberOfTimeStep` | `T`, number of time steps |
| `norm`, `norm_static` | normalisation of the temporal and static blocks |
| `use_static` | **must be declared explicitly**: `true` for the multimodal model, `false` for the temporal-only variant |
| `way_to_build_graph` | graph estimation method (`hgd`) |
| `n_trials` | number of Optuna trials |
| `train_params` | epochs, early stopping, and the ranges of learning rate and weight decay |

`use_static` has no default value: a missing key raises an error rather than
silently building a temporal-only model.

Splits and seeds are paired by position, so every architecture is trained and
evaluated on exactly the same ten partitions with the same initialisations,
which is what makes the paired statistical comparison in the paper valid.

---

## Notes on reproducibility

**Weight initialisation.** `StandardGCNNLayer` calls `torch.manual_seed` when it
is constructed. This makes runs deterministic and is the behaviour that produced
the reported results, but it also reseeds the global generator, so layers with
identical shapes receive identical initial weights.

**Optuna budget.** All models use 100 trials per split, declared in the
`n_trials` field of each `load_config.json`.

**Hardware.** Some numerical differences across GPU models have been observed in
weight initialisation. Results in the paper were produced on a single CUDA
device, configured through the `device` field of the configuration file.

---

## Licence

MIT — see `LICENSE`.
