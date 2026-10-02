"""MVPO instance and its Ising cost Hamiltonian."""

import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from portfolio_qaoa.config import DATA_DIR, Config


def bitstring_to_x(bitstring: str) -> tuple[int, ...]:
    """Qiskit bitstring -> portfolio x with x[i] = qubit i = ticker[i].

    Qiskit reads from the right: last character = qubit 0.
    """
    return tuple(int(b) for b in reversed(bitstring))


@dataclass
class Portfolio:
    ticker: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    budget: int
    lam: float

    @property
    def n(self) -> int:
        return len(self.ticker)

    def cost(self, x) -> float:
        """Portfolio objective C(x) = (1-λ) xᵀΣx - λ μᵀx."""
        x = np.asarray(x)
        return float((1 - self.lam) * x @ self.sigma @ x - self.lam * self.mu @ x)

    @cached_property
    def all_x(self) -> list[tuple[int, ...]]:
        return list(itertools.product([0, 1], repeat=self.n))

    @cached_property
    def feasible(self) -> list[tuple[int, ...]]:
        """Portfolios satisfying the budget, sorted by cost (the best first)."""
        return sorted((x for x in self.all_x if sum(x) == self.budget),
                      key=self.cost)

    @cached_property
    def optimum(self) -> tuple[int, ...]:
        """Best feasible portfolio (brute force)."""
        return self.feasible[0]

    @cached_property
    def alpha(self) -> float:
        """Penalty factor (Hodson et al. 2019): max C - min C + 0.01."""
        values = [self.cost(x) for x in self.all_x]
        return max(values) - min(values) + 0.01

    def name(self, x) -> str:
        """Portfolio as a string of ticker symbols, e.g., 'GOOG+NVDA'."""
        return "+".join(t for t, xi in zip(self.ticker, x, strict=True) if xi) or "–"


def load_portfolio(cfg: Config) -> Portfolio:
    data = np.load(DATA_DIR / cfg.datafile)
    return Portfolio(ticker=data["ticker"], mu=data["mu"], sigma=data["sigma"],
                     budget=cfg.budget, lam=cfg.lam)


# %% QUBO -> Ising cost Hamiltonian
def cost_hamiltonian(pf: Portfolio) -> tuple[SparsePauliOp, float]:
    qp = QuadraticProgram("portfolio")
    for t in pf.ticker:
        qp.binary_var(str(t))
    qp.minimize(linear=-pf.lam * pf.mu, quadratic=(1 - pf.lam) * pf.sigma)
    qp.linear_constraint(linear=[1] * pf.n, sense="==", rhs=pf.budget)
    qubo = QuadraticProgramToQubo(penalty=pf.alpha).convert(qp)
    h_c, offset = qubo.to_ising()
    check_hamiltonian(pf, h_c, offset)
    return h_c, offset


def check_hamiltonian(pf: Portfolio, h_c: SparsePauliOp, offset: float) -> None:
    """Diagonal of H_C + offset must equal C(x) + α(Σx - B)² for every x."""
    diag = np.real(np.diag(h_c.to_matrix())) + offset
    for k in range(2 ** pf.n):
        x = bitstring_to_x(format(k, f"0{pf.n}b"))
        expected = pf.cost(x) + pf.alpha * (sum(x) - pf.budget) ** 2
        if not np.isclose(diag[k], expected):
            raise ValueError(
                f"H_C does not match the QUBO for x = {x}: {diag[k]} != {expected}"
            )
