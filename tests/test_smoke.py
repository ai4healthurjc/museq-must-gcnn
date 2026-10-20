"""Smoke test for the two proposed architectures.

Builds MuSeq-GCNN and MuST-GCNN on synthetic data of the shapes used in the
paper, runs a forward pass and one optimisation step, and checks the shapes and
ranges of every quantity the rest of the pipeline relies on.

Run with:  python -m pytest tests/test_smoke.py -v
or simply: python tests/test_smoke.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "utils"))

from architectures.MuSeqGCNN.model import MultimodalGNN            # noqa: E402
from architectures.MuSTGCNN.model import MultimodalGNN_ST          # noqa: E402

# Shapes of the AMR task (F = 85 temporal variables, T = 14 steps, D = 3 static)
F, T, D, B = 85, 14, 3, 8
H, HS, HR = 32, 32, 50


def _random_adjacency(n, density=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g) < density).float()
    a = torch.triu(a, diagonal=1)
    return a + a.T


def _gnn_params(in_dim, out_dim, layers=2, seed=1):
    return {"in_dim": in_dim, "out_dim": out_dim, "hid_dim": 16,
            "layers": layers, "dropout": 0.1, "seed": seed,
            "embedding_dim": out_dim}


def test_museq_forward_and_step():
    S_temporal = [_random_adjacency(F, seed=t) for t in range(T)]
    S_static = _random_adjacency(D, density=0.5, seed=99)
    model = MultimodalGNN(
        gnn_params_temporal=_gnn_params(1, H),
        gnn_params_static=_gnn_params(1, HS),
        rnn_params={"hidden_dim": HR, "num_layers": 1, "bidirectional": False,
                    "dropout": 0.0},
        S_temporal=S_temporal,
        S_static=S_static,
        use_static=True,
    )
    x_temporal = [torch.randn(B, F, 1) for _ in range(T)]
    x_static = torch.randn(B, D, 1)

    out = model(x_temporal, x_static)
    assert out["probability"].shape == (B, 1)
    assert out["attention_alpha"].shape == (B, 1)
    assert torch.all((out["attention_alpha"] > 0) & (out["attention_alpha"] < 1))
    assert torch.all((out["probability"] >= 0) & (out["probability"] <= 1))

    loss = torch.nn.functional.binary_cross_entropy(
        out["probability"], torch.randint(0, 2, (B, 1)).float())
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    print("MuSeq-GCNN OK   alpha mean =", float(out["attention_alpha"].detach().mean()))


def test_must_forward_and_step():
    S_st = _random_adjacency(F * T, density=0.01, seed=7)
    S_static = _random_adjacency(D, density=0.5, seed=99)
    model = MultimodalGNN_ST(
        gnn_params_temporal=_gnn_params(1, H),
        gnn_params_static=_gnn_params(1, HS),
        S_temporal=S_st,
        S_static=S_static,
        use_static=True,
    )
    x_st = torch.randn(B, F * T, 1)
    x_static = torch.randn(B, D, 1)

    out = model(x_st, x_static)
    assert out["probability"].shape == (B, 1)
    assert out["temporal_embedding"].shape == (B, F * T)   # one scalar per (f, t)
    assert out["static_embedding"].shape == (B, D)          # one scalar per static node
    assert out["attention_alpha"].shape == (B, 1)
    assert torch.all((out["attention_alpha"] > 0) & (out["attention_alpha"] < 1))

    loss = torch.nn.functional.binary_cross_entropy(
        out["probability"], torch.randint(0, 2, (B, 1)).float())
    loss.backward()
    print("MuST-GCNN OK    alpha mean =", float(out["attention_alpha"].detach().mean()))


def test_temporal_only_branch():
    """use_static=False must produce a working temporal-only model."""
    S_temporal = [_random_adjacency(F, seed=t) for t in range(T)]
    S_static = _random_adjacency(D, density=0.5, seed=99)
    model = MultimodalGNN(
        gnn_params_temporal=_gnn_params(1, H),
        gnn_params_static=_gnn_params(1, HS),
        rnn_params={"hidden_dim": HR, "num_layers": 1, "bidirectional": False,
                    "dropout": 0.0},
        S_temporal=S_temporal,
        S_static=S_static,
        use_static=False,
    )
    out = model([torch.randn(B, F, 1) for _ in range(T)], torch.randn(B, D, 1))
    assert out["attention_alpha"] is None
    assert out["probability"].shape == (B, 1)
    print("temporal-only OK")


if __name__ == "__main__":
    test_museq_forward_and_step()
    test_must_forward_and_step()
    test_temporal_only_branch()
    print("\nAll smoke tests passed.")
