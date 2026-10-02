"""Figures for the slides, including the gate-level circuit drawing."""

import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Parameter
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


def _ordered_cost_terms(h_c: SparsePauliOp) -> list[tuple[list[int], float]]:
    """Cost terms of H_C as (qubits, coeff), ordered for a narrow figure.

    The terms are all diagonal and commute, so they can be reordered freely:
    all Rz first, then the ZZ gates packed into as few columns as possible
    (e.g. ZZ(0,1) and ZZ(2,3) side by side).
    """
    terms = [(np.flatnonzero(pauli.z).tolist(), coeff)
             for pauli, coeff in zip(h_c.paulis, h_c.coeffs.real, strict=True)]
    singles = [t for t in terms if len(t[0]) == 1]
    # Greedy packing: a ZZ gate joins the first column whose gates' wire spans
    # it does not overlap (the vertical line of ZZ(i, j) covers wires i…j)
    columns: list[list] = []
    for t in sorted((t for t in terms if len(t[0]) == 2), key=lambda t: t[0]):
        i, j = t[0]
        col = next((c for c in columns
                    if all(j < a or b < i for (a, b), _ in c)), None)
        if col is None:
            columns.append(col := [])
        col.append(t)
    return singles + [t for col in columns for t in col]


def _append_cost_layer(qc: QuantumCircuit, terms: list, gamma: Parameter,
                       ks: str, symbolic: bool) -> None:
    """Append U_C(γ): Z_i -> Rz(2h_i·γ), Z_iZ_j -> Rzz(2J_ij·γ)."""
    sep = "" if qc.num_qubits < 10 else ","  # J_12, or J_{10,11} for many qubits
    for qubits, coeff in terms:
        if symbolic:
            idx = sep.join(str(q + 1) for q in qubits)
            name = "h" if len(qubits) == 1 else "J"
            angle = Parameter(rf"$2{name}_{{{idx}}}\gamma_{ks}$")
        else:
            angle = 2 * coeff * gamma
        if len(qubits) == 1:
            qc.rz(angle, qubits[0])
        else:
            qc.rzz(angle, *qubits)


def drawing_circuit(h_c: SparsePauliOp, reps: int,
                    symbolic: bool = True) -> QuantumCircuit:
    """Gate-level QAOA circuit with labeled layers, for display only.

    Same gates as qaoa_ansatz for H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j:
    Z_i -> Rz(2h_i·γ), Z_iZ_j -> Rzz(2J_ij·γ), mixer Rx(2β).
    symbolic=True labels the angles with h_i, J_ij (no numbers);
    symbolic=False uses the exact coefficients (for checking against qaoa_ansatz).
    Measurements are drawn as "M" boxes without classical bits, so the drawer
    places them in one column (real measurements are always staggered).
    The cost terms are reordered to make the figure narrower
    (see _ordered_cost_terms).
    """
    ordered = _ordered_cost_terms(h_c)

    # ASCII names; plot_circuit turns them into $x_{i}$ (Qiskit escapes "_")
    regs = [QuantumRegister(1, f"x{i + 1}") for i in range(h_c.num_qubits)]
    qc = QuantumCircuit(*regs)
    # Each barrier label marks the block that follows it;
    # the initial state is shown by the |0⟩ wire labels
    qc.h(range(qc.num_qubits))
    for k in range(1, reps + 1):
        # $...$ is rendered by matplotlib mathtext; braces only for k >= 10,
        # since Qiskit truncates barrier labels longer than 16 characters
        ks = str(k) if k < 10 else f"{{{k}}}"
        gamma = Parameter(rf"$\gamma_{ks}$")
        beta = Parameter(rf"$\beta_{ks}$")
        qc.barrier(label=rf"$U_C(\gamma_{ks})$")
        _append_cost_layer(qc, ordered, gamma, ks, symbolic)
        qc.barrier(label=rf"$U_M(\beta_{ks})$")
        mixer_angle = Parameter(rf"$2\beta_{ks}$") if symbolic else 2 * beta
        qc.rx(mixer_angle, range(qc.num_qubits))
    qc.barrier(label="Measurement")
    for q in range(qc.num_qubits):
        qc.append(Instruction("M", 1, 0, [], label=r"$\mathcal{M}$"), [q])
    return qc


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
    infeasible = sorted(set(pf.all_x) - set(pf.feasible), key=pf.cost)
    x_sorted = pf.feasible + infeasible
    categories = {"tab:green": "Optimum",
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
    # Bar colours are not labelled artists, so their legend entries are
    # added as patches in front of the guessing line
    handles = [Patch(color=c, label=label) for c, label in categories.items()]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="lower right")
    _save(fig, path)


def _adjust_circuit_axes(ax, box_height: float = 0.9, width_pad: float = 1.1,
                         small_size: float = 0.7,
                         label_shift: float = 0.2) -> None:
    """Post-process Qiskit's circuit figure (not configurable in its drawer).

    - Rotation gates (Rz, Rx): `box_height` tall (Qiskit: fixed 0.65, wire
      spacing = 1) and `width_pad` × the widest of them wide, so the two-line
      labels fit; the label lines move apart.
    - H and M (one-line labels): smaller squares of side `small_size`.
    - All boxes stay centred where Qiskit placed them.
    - Barrier labels (U_C, U_M, Measurement): moved `label_shift` further up.
    """

    def is_rotation(s: str) -> bool:
        return s.startswith(r"$\mathrm{R_")

    def is_small(s: str) -> bool:
        return s in ("H", r"$\mathcal{M}$")

    rotations, smalls = [], []  # (box, texts inside it)
    for box in [p for p in ax.patches if type(p) is Rectangle]:
        x0, y0 = box.get_xy()
        w, h = box.get_width(), box.get_height()
        inside = [t for t in ax.texts
                  if x0 <= t.get_position()[0] <= x0 + w
                  and y0 <= t.get_position()[1] <= y0 + h]
        labels = [t.get_text() for t in inside]
        if any(map(is_rotation, labels)):
            rotations.append((box, inside))
        elif any(map(is_small, labels)):
            smalls.append((box, inside))
    rot_width = width_pad * max((b.get_width() for b, _ in rotations), default=0)
    for group, (width, height) in ((rotations, (rot_width, box_height)),
                                   (smalls, (small_size, small_size))):
        for box, inside in group:
            x0, y0 = box.get_xy()
            xc = x0 + box.get_width() / 2
            yc = y0 + box.get_height() / 2
            factor = height / box.get_height()
            box.set_bounds(xc - width / 2, yc - height / 2, width, height)
            for t in inside:
                t.set_y(yc + (t.get_position()[1] - yc) * factor)
    for t in ax.texts:
        if t.get_text().startswith("$U_") or t.get_text() == "Measurement":
            t.set_y(t.get_position()[1] + label_shift)


def plot_circuit(h_c: SparsePauliOp, reps: int, ticker, path: Path) -> None:
    """Gate-level QAOA circuit: |0⟩, H, cost layer, mixer, measurement."""
    qc = drawing_circuit(h_c, reps)
    style = {"name": "iqp", "displaycolor": {"M": ("#A0A0A0", "#000000")}}
    # Computer Modern (LaTeX font) for all $...$ math
    with plt.rc_context({"mathtext.fontset": "cm"}):
        # Qiskit sizes the figure from the gate layout (about 22 in for p = 1),
        # not from the font size. Scale the fonts by width / slide width so
        # they reach the CIRCUIT_*_PT sizes when shown at slide width.
        probe = qc.draw("mpl", initial_state=True, fold=-1, style=style)
        # width after bbox_inches="tight" (Qiskit adds white margins)
        k = probe.get_tightbbox().width / SLIDE_FIG_WIDTH
        plt.close(probe)
        style |= {"fontsize": CIRCUIT_GATE_PT * k,
                  "subfontsize": CIRCUIT_PARAM_PT * k}
        fig = qc.draw("mpl", initial_state=True, fold=-1, style=style)
        # Qiskit labels the wires "${x1}$ $|0\rangle$" -> "$x_{1}$ $∣0⟩$".
        # mathtext's \rangle is too big next to "|"; the Unicode glyphs
        # ∣ (U+2223) and ⟩ (U+27E9) have equal height in Computer Modern,
        # like \left|0\right\rangle in LaTeX
        for text in fig.axes[0].texts:
            label = re.sub(r"^\$\{x(\d+)}\$", r"$x_{\1}$", text.get_text())
            text.set_text(label.replace(r"$|0\rangle$", "$∣0⟩$"))
        _adjust_circuit_axes(fig.axes[0])
        mapping = ", ".join(rf"$x_{{{i}}}$={t}"
                            for i, t in enumerate(ticker, start=1))
        fig.suptitle(rf"QAOA circuit – {mapping}",
                     fontsize=PRESENTATION_RC["figure.titlesize"] * k)
        fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
