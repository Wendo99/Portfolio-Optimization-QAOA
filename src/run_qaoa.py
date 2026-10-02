"""QAOA for mean-variance portfolio optimization (MVPO) with a budget constraint."""

import numpy as np
from matplotlib import pyplot as plt

from portfolio_qaoa.config import ROOT, Config
from portfolio_qaoa.plots import (PRESENTATION_RC, plot_circuit, plot_convergence,
                                  plot_distribution, plot_landscape)
from portfolio_qaoa.qaoa import run
from portfolio_qaoa.report import analyze, print_summary, save_results


def _tagged(stem: str, tag: str) -> str:
    """Output name for a run variant, e.g. convergence_p2."""
    return f"{stem}_{tag}" if tag else stem


def main(cfg: Config | None = None, tag: str = "") -> dict:
    """Run QAOA and write results/results[_tag].json and images/*[_tag].png."""
    cfg = cfg or Config()
    r = run(cfg)
    print(f"Starting point from grid search: β = {r.grid.beta0:.3f}, γ = {r.grid.gamma0:.3f}")
    betas_opt, gammas_opt = r.result.x[:cfg.reps], r.result.x[cfg.reps:]
    print(f"Optimized: γ* = {np.round(gammas_opt, 4)}, β* = {np.round(betas_opt, 4)}, "
          f"F_C = {r.result.fun:.4f} after {len(r.history)} evaluations")

    summary = analyze(r.pf, r.prob)
    print_summary(r.pf, summary)
    save_results(ROOT / "results" / f"{_tagged('results', tag)}.json", r, summary)

    images = ROOT / "images"
    images.mkdir(exist_ok=True)

    def image(stem: str):
        return images / f"{_tagged(stem, tag)}.png"

    with plt.rc_context(PRESENTATION_RC):
        plot_convergence(r.history, cfg.reps, image("convergence"))
        plot_landscape(r.grid, image("energy_landscape"))
        plot_distribution(r.pf, r.prob, cfg, image("measurement_distribution"))
        plot_circuit(r.h_c, cfg.reps, r.pf.ticker, image("qaoa_circuit"))
    return summary


if __name__ == "__main__":
    main()
