"""Analysis of the measurement distribution and results.json."""

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import qiskit
import qiskit_optimization
import scipy

from portfolio_qaoa.config import ROOT
from portfolio_qaoa.problem import Portfolio
from portfolio_qaoa.qaoa import RunResult


def analyze(pf: Portfolio, prob: dict) -> dict:
    x_opt = pf.optimum
    return {
        "optimum": pf.name(x_opt),
        "P_feas": sum(prob[x] for x in pf.feasible),
        "P_opt": prob[x_opt],
        "P_feas_random": len(pf.feasible) / 2 ** pf.n,
        "P_opt_random": 1 / 2 ** pf.n,
        "feasible_portfolios": {
            pf.name(x): {"C": pf.cost(x), "measured": prob[x]}
            for x in pf.feasible
        },
        "infeasible_portfolios": {
            pf.name(x): {"C": pf.cost(x), "measured": prob[x]}
            for x in sorted(set(pf.all_x) - set(pf.feasible), key=pf.cost)
        },
    }


def print_summary(pf: Portfolio, summary: dict) -> None:
    print(f"Optimum (Brute Force): {summary['optimum']}")
    print(f"P_feas = {summary['P_feas']:.1%}   "
          f"(random: {len(pf.feasible)}/{2 ** pf.n} = {summary['P_feas_random']:.1%})")
    print(f"P_opt  = {summary['P_opt']:.1%}   "
          f"(random: 1/{2 ** pf.n} = {summary['P_opt_random']:.1%})")
    print("\nFeasible portfolios, sorted by cost:")
    for label, row in summary["feasible_portfolios"].items():
        print(f"  {label:10s} C = {row['C']:+.4f}   measured: {row['measured']:.1%}")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_results(path: Path, r: RunResult, summary: dict) -> None:
    """Everything needed to reproduce and compare a run, and the numbers for the slides."""
    reps, pf, result = r.cfg.reps, r.pf, r.result
    status = _git("status", "--porcelain")
    results = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parameter": {**asdict(r.cfg), "n": pf.n, "alpha": pf.alpha},
        "instance": {"ticker": pf.ticker.tolist(), "mu": pf.mu.tolist(),
                     "sigma": pf.sigma.tolist()},
        "grid_search": {"beta0": r.grid.beta0, "gamma0": r.grid.gamma0,
                        "expected_cost_min": r.grid.raster.min()},
        "optimization": {"beta_opt": result.x[:reps].tolist(),
                         "gamma_opt": result.x[reps:].tolist(),
                         "expected_cost": result.fun,
                         "cobyla_evaluations": len(r.history),
                         "success": bool(result.success),
                         "message": result.message,
                         "history": r.history},
        "result": summary,
        "versions": {"python": sys.version.split()[0],
                     "qiskit": qiskit.__version__,
                     "qiskit_optimization": qiskit_optimization.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "git_commit": _git("rev-parse", "HEAD"),
                     "git_dirty": None if status is None else bool(status)},
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
