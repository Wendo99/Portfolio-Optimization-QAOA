# QAOA – Portfolio Optimization Demo

WORK IN PROGRESS

A technical demonstration mapping a classical financial optimization problem (Markowitz
mean-variance) to a combinatorial Quadratic Unconstrained Binary Optimization (QUBO) problem, solved
via the Quantum Approximate Optimization Algorithm (QAOA) using IBM Qiskit.

The project explores how a classical financial optimization problem can be transformed into a
combinatorial optimization problem and subsequently formulated for a quantum optimization algorithm.

## Overview

Portfolio optimization aims to find an allocation that balances expected return and risk.
Historical market data is obtained using [`yfinance`](https://github.com/ranaroussi/yfinance).

The optimization objective combines expected returns and portfolio risk and is formulated in a
Markowitz mean-variance model. The resulting Quadratic Unconstrained Binary Optimization (QUBO) is
then mapped to an Ising Hamiltonian and used as the cost Hamiltonian for QAOA.

The project therefore follows the workflow:

```text
Market Data (yfinance)
    ↓
Expected Returns & Covariance Matrix
    ↓
Markowitz Mean-Variance Model (Budget constraint)
    ↓
QUBO Formulation (Penalty factor α)
    ↓
Ising Hamiltonian (Cost operator H_C)
    ↓
QAOA Circuit (Statevector Simulation & COBYLA)
    ↓
Measurement & Evaluation
```

## Results

The model selects exactly 2 out of 4 assets (GOOG, AMZN, META, NVDA) based on 2025 historical data.
The exact analytical optimum is GOOG+NVDA.

The QAOA implementation successfully shifts the measurement distribution away from random guessing
(6.25% for the optimal state) towards the feasible optimum:
Depth $p=1$: Measured probability of the optimal portfolio: 56.6%
Depth $p=2$: Measured probability of the optimal portfolio: 62.7% 

(Parameters optimized via COBYLA on a deterministic statevector simulator).

## Project Structure

The project is structured as a modular Python package (portfolio_qaoa).

- `data/`: Frozen historical market data (.csv and .npz) to guarantee deterministic execution.
- `src/portfolio_qaoa/`: The core package module containing the financial model, QUBO conversion, QAOA execution, and reporting logic.
- `src/run_data.py`: Script for data acquisition and annualization.
- `src/run_qaoa.py`: Main execution script for the quantum optimization pipeline.
- `results/`: Tracked JSON outputs of optimization runs (includes Git commit hashes for reproducibility).
- `images/`: Generated matplotlib figures (convergence, landscapes, circuit diagrams).

## Scope and Honest Limitations

This is purely an educational demonstration of encoding binary asset-selection into a QUBO and
sampling it via QAOA.

- No Quantum Advantage: The default instance handles 4 assets (16 total states) and is simulated
  classically. The exact feasible optimum is trivially found by enumeration.
- Simplified Financial Model: It uses historical estimates from a single year and does not model
  continuous asset weights, share counts, transaction costs, or trading constraints.
- Disclaimer: This project does not constitute financial advice or an investment recommendation.

## Installation and Run

Built with Python 3.14+ and Qiskit 2.5+.

Clone the repository:

```bash
# Clone the repository
git clone [https://github.com/Wendo99/Portfolio_Optimization_Demo.git](https://github.com/Wendo99/Portfolio_Optimization_Demo.git)
cd Portfolio_Optimization_Demo

# Install dependencies via uv
uv sync

# Run the optimization pipeline
uv run python src/02_portfolio_optimization.py
```