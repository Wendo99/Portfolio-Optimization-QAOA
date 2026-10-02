"""QAOA energy evaluation, parameter search and sampling."""

from dataclasses import dataclass

import numpy as np
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import OptimizeResult, minimize

from portfolio_qaoa.config import Config
from portfolio_qaoa.problem import (Portfolio, bitstring_to_x, cost_hamiltonian,
                                    load_portfolio)


class QAOA:
    """Energy evaluation for a QAOA ansatz.

    Parameter vectors follow ansatz.parameters: (β_1…β_p, γ_1…γ_p).
    """

    def __init__(self, h_c: SparsePauliOp, offset: float, reps: int,
                 estimator: StatevectorEstimator):
        self.h_c = h_c
        self.offset = offset
        self.reps = reps
        self.ansatz = qaoa_ansatz(cost_operator=h_c, reps=reps)
        self.estimator = estimator
        self.history: list[float] = []  # <C> per COBYLA evaluation

    def energy(self, params) -> np.ndarray:
        """<ψ|H_C|ψ> + offset = expected cost C; params may be batched (..., 2p)."""
        pub = (self.ansatz, self.h_c, np.asarray(params))
        return self.estimator.run([pub]).result()[0].data.evs + self.offset

    def _objective(self, params) -> float:
        value = float(self.energy(params))
        self.history.append(value)
        return value

    def optimize(self, x0, maxiter: int, rhobeg: float) -> OptimizeResult:
        self.history.clear()
        return minimize(self._objective, x0=x0, method="COBYLA",
                        options={"maxiter": maxiter, "rhobeg": rhobeg})


@dataclass
class GridSearch:
    """p = 1 energy landscape raster[β, γ] and its minimum (β0, γ0)."""
    betas: np.ndarray
    gammas: np.ndarray
    raster: np.ndarray
    beta0: float
    gamma0: float


def grid_search(qaoa_p1: QAOA, cfg: Config) -> GridSearch:
    """Rough search over the p = 1 landscape.

    β is π-periodic. γ is NOT periodic because the coefficients of H_C are
    not integers -> a large search window is required.
    (γ < 0 yields no new results due to the symmetry (γ, β) -> (-γ, -β).)
    """
    betas = np.linspace(0, np.pi, cfg.n_beta, endpoint=False)
    gammas = np.linspace(0, cfg.gamma_max, cfg.n_gamma)
    bb, gg = np.meshgrid(betas, gammas, indexing="ij")
    raster = qaoa_p1.energy(np.stack([bb, gg], axis=-1))  # one batched call
    ib, ig = np.unravel_index(raster.argmin(), raster.shape)
    return GridSearch(betas, gammas, raster, betas[ib], gammas[ig])


def initial_point(beta0: float, gamma0: float, reps: int) -> np.ndarray:
    """Repeat the p = 1 optimum in every layer (= INTERP for p = 2, Zhou et al. 2020)."""
    return np.concatenate([np.full(reps, beta0), np.full(reps, gamma0)])


def sample(qaoa: QAOA, params, pf: Portfolio,
           sampler: StatevectorSampler, shots: int) -> dict:
    circuit = qaoa.ansatz.assign_parameters(params)
    circuit.measure_all()
    counts = sampler.run([circuit], shots=shots).result()[0].data.meas.get_counts()
    prob = dict.fromkeys(pf.all_x, 0.0)
    for bitstring, num in counts.items():
        prob[bitstring_to_x(bitstring)] = num / shots
    return prob


@dataclass
class RunResult:
    cfg: Config
    pf: Portfolio
    h_c: SparsePauliOp
    offset: float
    grid: GridSearch
    result: OptimizeResult  # x = (β_1…β_p, γ_1…γ_p), fun = expected cost
    history: list[float]  # <C> per COBYLA evaluation
    prob: dict  # measured probability per portfolio x


def run(cfg: Config) -> RunResult:
    """Grid search at p = 1, COBYLA at p = cfg.reps, then sampling.

    No output: printing and saving are left to the caller, so runs can be
    repeated freely, e.g. [run(replace(cfg, lam=lam)) for lam in lams].
    """
    pf = load_portfolio(cfg)
    h_c, offset = cost_hamiltonian(pf)
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler(seed=cfg.seed)

    # 1. Rough search on the p = 1 landscape
    qaoa_p1 = QAOA(h_c, offset, reps=1, estimator=estimator)
    grid = grid_search(qaoa_p1, cfg)

    # 2. Classical optimizer: COBYLA
    qaoa = qaoa_p1 if cfg.reps == 1 else QAOA(h_c, offset, cfg.reps, estimator)
    result = qaoa.optimize(initial_point(grid.beta0, grid.gamma0, cfg.reps),
                           cfg.maxiter, cfg.rhobeg)

    # 3. Measuring with the optimal parameters
    prob = sample(qaoa, result.x, pf, sampler, cfg.shots)
    return RunResult(cfg, pf, h_c, offset, grid, result, list(qaoa.history), prob)
