"""Analysis of the measurement distribution and results.json."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import qiskit
import qiskit_optimization
import scipy

from portfolio_qaoa.config import Config
from portfolio_qaoa.problem import Portfolio


def analyze(pf: Portfolio, prob: dict) -> dict:
    x_opt = pf.optimum
    return {
        "optimum": pf.name(x_opt),
        "P_zul": sum(prob[x] for x in pf.feasible),
        "P_opt": prob[x_opt],
        "P_zul_raten": len(pf.feasible) / 2 ** pf.n,
        "P_opt_raten": 1 / 2 ** pf.n,
        "zulaessige_portfolios": {
            pf.name(x): {"C": pf.cost(x), "gemessen": prob[x]}
            for x in pf.feasible
        },
    }


def print_summary(pf: Portfolio, summary: dict) -> None:
    print(f"Optimum (Brute Force): {summary['optimum']}")
    print(f"P_zul = {summary['P_zul']:.1%}   "
          f"(Raten: {len(pf.feasible)}/{2 ** pf.n} = {summary['P_zul_raten']:.1%})")
    print(f"P_opt = {summary['P_opt']:.1%}   "
          f"(Raten: 1/{2 ** pf.n} = {summary['P_opt_raten']:.1%})")
    print("\nZulässige Portfolios, sortiert nach Kosten:")
    for label, row in summary["zulaessige_portfolios"].items():
        print(f"  {label:10s} C = {row['C']:+.4f}   gemessen: {row['gemessen']:.1%}")


def save_results(path: Path, cfg: Config, pf: Portfolio, result, n_evals: int,
                 summary: dict) -> None:
    """Source for all numbers on the slides."""
    reps = cfg.reps
    results = {
        "parameter": {**asdict(cfg), "n": pf.n, "alpha": pf.alpha},
        "optimierung": {"beta_opt": result.x[:reps].tolist(),
                        "gamma_opt": result.x[reps:].tolist(),
                        "erwartete_kosten": result.fun,
                        "auswertungen_cobyla": n_evals},
        "ergebnis": summary,
        "versionen": {"python": sys.version.split()[0],
                      "qiskit": qiskit.__version__,
                      "qiskit_optimization": qiskit_optimization.__version__,
                      "numpy": np.__version__, "scipy": scipy.__version__},
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
