"""QAOA for mean-variance portfolio optimization (MVPO) with a budget constraint."""

import itertools
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from portfolio_qaoa.config import ROOT, Config
from portfolio_qaoa.plots import (PRESENTATION_RC, plot_circuit, plot_convergence,
                                  plot_distribution, plot_landscape)
from portfolio_qaoa.qaoa import run
from portfolio_qaoa.report import analyze, print_summary, save_results


IMAGES = ("convergence", "energy_landscape", "measurement_distribution", "qaoa_circuit")


def _tagged(stem: str, tag: str) -> str:
    """Output name for a run variant, e.g. convergence_p2."""
    return f"{stem}_{tag}" if tag else stem


def _output_paths(tag: str) -> tuple[Path, dict[str, Path]]:
    """results.json and image paths for a run that overwrite nothing.

    If any output of `tag` already exists, a counter is appended to all of
    them together (p2 -> p2_2, p2_3, …; no tag -> 2, 3, …), so the JSON and
    its images always share one name.
    """
    for k in itertools.count(1):
        run_tag = tag if k == 1 else _tagged(tag, str(k)) if tag else str(k)
        json_path = ROOT / "results" / f"{_tagged('results', run_tag)}.json"
        images = {stem: ROOT / "images" / f"{_tagged(stem, run_tag)}.png"
                  for stem in IMAGES}
        if not any(p.exists() for p in (json_path, *images.values())):
            return json_path, images


def main(cfg: Config | None = None, tag: str = "") -> dict:
    """Run QAOA and write results/results[_tag][_k].json and images/*[_tag][_k].png.

    Existing files are never overwritten: see _output_paths for the counter k.
    """
    cfg = cfg or Config()
    r = run(cfg)
    print(f"Starting point from grid search: β = {r.grid.beta0:.3f}, γ = {r.grid.gamma0:.3f}")
    betas_opt, gammas_opt = r.result.x[:cfg.reps], r.result.x[cfg.reps:]
    print(f"Optimized: γ* = {np.round(gammas_opt, 4)}, β* = {np.round(betas_opt, 4)}, "
          f"F_C = {r.result.fun:.4f} after {len(r.history)} evaluations")

    summary = analyze(r.pf, r.prob)
    print_summary(r.pf, summary)
    json_path, images = _output_paths(tag)
    save_results(json_path, r, summary)

    (ROOT / "images").mkdir(exist_ok=True)
    with plt.rc_context(PRESENTATION_RC):
        plot_convergence(r.history, cfg.reps, images["convergence"])
        plot_landscape(r.grid, images["energy_landscape"])
        plot_distribution(r.pf, r.prob, cfg, images["measurement_distribution"])
        plot_circuit(r.h_c, cfg.reps, r.pf.ticker, images["qaoa_circuit"])
    print(f"\nSaved: {json_path.name}, " + ", ".join(p.name for p in images.values()))
    return summary


if __name__ == "__main__":
    main(tag="p2")
