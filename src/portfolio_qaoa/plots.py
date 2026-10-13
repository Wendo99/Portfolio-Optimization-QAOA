"""Figures for the slides, including the gate-level circuit drawing."""

import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp

from portfolio_qaoa.config import Config
from portfolio_qaoa.problem import Portfolio
from portfolio_qaoa.qaoa import GridSearch

# Presentation fonts. Figures are 12 in wide, i.e. about the content width of
# a 16:9 slide (13.33 in), so these point sizes are what the audience sees
# when a plot is placed at full slide width.
SLIDE_FIG_WIDTH = 12.0
PRESENTATION_RC = {
    "font.size": 16,
    "axes.titlesize": 22,
    "figure.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}
# Circuit plot: point sizes on the slide for gate names and gate angles.
# Smaller than the other plots: Qiskit's layout does not grow with the font,
# and with ~22 gate columns a column is only ~0.5 in wide at slide width.
CIRCUIT_GATE_PT = 13
CIRCUIT_PARAM_PT = 11


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def plot_convergence(history: list[float], reps: int, path: Path) -> None:
    """Convergence plot (according to Stein)."""
    fig, ax = plt.subplots(figsize=(SLIDE_FIG_WIDTH, 5))
    ax.plot(history, marker=".")
    ax.set_xlabel("Iteration (COBYLA)")
    ax.set_ylabel(r"$F_p$ – expected cost")
    ax.set_title(rf"Convergence of the parameter optimization $(p = {reps})$")
    _save(fig, path)


def plot_landscape(grid: GridSearch, path: Path) -> None:
    """Energy landscape of the p = 1 grid search (backup slide)."""
    fig, ax = plt.subplots(figsize=(SLIDE_FIG_WIDTH, 5))
    mesh = ax.pcolormesh(grid.gammas, grid.betas, grid.raster, shading="nearest",
                         cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=r"$F_p$ – expected cost")
    ax.plot(grid.gamma0, grid.beta0, marker="*", color="red", markersize=14,
            label="Minimum (grid)")
    ax.axvline(2 * np.pi, color="white", linestyle="--",
               label=r"$\gamma = 2\pi$ (usual search window)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Energy landscape $F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})$, $p = 1$")
    ax.legend(loc="upper right", fontsize="small")
    _save(fig, path)


def plot_distribution(pf: Portfolio, prob: dict, cfg: Config, path: Path) -> None:
    """Measurement distribution: feasible portfolios first, sorted by cost."""
    x_sorted = pf.feasible + pf.infeasible
    categories = {"tab:green": "optimum",
                  "tab:blue": "feasible",
                  "tab:red": "infeasible"}
    feasible = set(pf.feasible)
    colors = ["tab:green" if x == pf.optimum
              else "tab:blue" if x in feasible
    else "tab:red" for x in x_sorted]
    # Horizontal bars: the 16 portfolio names stay readable at presentation
    # font size (vertical bars would need rotated or overlapping labels)
    fig, ax = plt.subplots(figsize=(SLIDE_FIG_WIDTH, 6.5))
    ax.barh([pf.name(x) for x in x_sorted], [prob[x] for x in x_sorted],
            color=colors)
    ax.invert_yaxis()  # best portfolio at the top
    ax.axvline(1 / 2 ** pf.n, color="gray", linestyle="--",
               label=rf"Random guess $(1/{2 ** pf.n})$")
    ax.set_xlabel("Measurement probability")
    # Centred on the figure, not the axes (the long labels shift the axes
    # right). {{\min}}: literal braces; a bare {min} would insert Python's min()
    fig.suptitle(f"Measurement distribution after optimization "
                 rf"($p = {cfg.reps}$, $\alpha_{{\min}}$, {cfg.shots} shots)")
    # Bar colors are not labeled artists, so their legend entries are
    # added as patches in front of the guessing line
    handles = [Patch(color=c, label=label) for c, label in categories.items()]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="lower right")
    _save(fig, path)


def plot_circuit(h_c: SparsePauliOp, reps: int, ticker, path: Path) -> None:
    """Gate-level QAOA circuit: |0⟩, H, cost layer, mixer, measurement."""
    qc = qaoa_ansatz(h_c, reps)
    qc.measure_all()
    style = {"name": "iqp", "displaycolor": {"M": ("#A0A0A0", "#000000")}}
    fig = qc.draw("mpl", initial_state=True, fold=-1, style=style, justify="left",
                  vertical_compression="high", idle_wires=False)
    mapping = ", ".join(rf"$x_{{{i}}}$={t}"
                        for i, t in enumerate(ticker, start=1))
    fig.suptitle(rf"QAOA circuit – {mapping}",
                 fontsize=PRESENTATION_RC["figure.titlesize"])
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
