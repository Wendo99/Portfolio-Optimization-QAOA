"""Paths and run parameters."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_DATAFILE = "portfolio_daten.npz"


@dataclass(frozen=True)
class Config:
    datafile: str = DEFAULT_DATAFILE

    # MVPO / QUBO
    budget: int = 2  # B
    lam: float = 0.5  # risk/return trade-off λ

    # Circuit
    reps: int = 1  # p

    # Grid search (always at p = 1): β ∈ [0, π), γ ∈ [0, gamma_max]
    gamma_max: float = 16 * np.pi
    n_gamma: int = 800
    n_beta: int = 40

    # COBYLA; rhobeg ≈ grid spacing (γ: 16π/799 ≈ 0.063, β: π/40 ≈ 0.079)
    rhobeg: float = 0.1
    maxiter: int = 2000

    # Simulation
    shots: int = 10_000
    seed: int = 42
