"""Price data and the portfolio instance (mu, sigma) derived from it."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_qaoa.config import DATA_DIR, DEFAULT_DATAFILE


@dataclass(frozen=True)
class DataConfig:
    ticker: tuple[str, ...] = ("GOOG", "AMZN", "META", "NVDA")
    start: str = "2024-12-31"
    end: str = "2026-01-01"  # end is exclusive in yfinance
    trading_days_per_year: int = 252
    datafile: str = DEFAULT_DATAFILE
    pricefile: str = "kurse_2025.csv"


def load_prices(cfg: DataConfig) -> pd.DataFrame:
    """Dividend-/split-adjusted closing prices, columns in cfg.ticker order."""
    downloaded = yf.download(list(cfg.ticker), start=cfg.start, end=cfg.end,
                             auto_adjust=True, progress=False)
    if downloaded is None:
        raise RuntimeError("yfinance returned no price data")
    prices = downloaded["Close"]
    return prices[list(cfg.ticker)]


def annualized_moments(returns: pd.DataFrame,
                       cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Annualization (Hodson et al. 2019): does not change the optimal solution,
    since risk and return term are scaled by the same factor, but gives
    sensible magnitudes for the angles in the circuit.
    """
    mu = returns.mean().to_numpy() * cfg.trading_days_per_year
    sigma = returns.cov().to_numpy() * cfg.trading_days_per_year
    return mu, sigma


def slide_table(prices: pd.DataFrame, returns: pd.DataFrame,
                cfg: DataConfig) -> pd.DataFrame:
    """Illustrative figures for slide 10."""
    return pd.DataFrame({
        "Rendite 2025": prices.iloc[-1] / prices.iloc[0] - 1,
        "Volatilität p.a.": returns.std() * np.sqrt(cfg.trading_days_per_year),
    })


def save(cfg: DataConfig, prices: pd.DataFrame, mu: np.ndarray,
         sigma: np.ndarray) -> None:
    """Freeze the data – all further steps only read these files."""
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(DATA_DIR / cfg.datafile, ticker=list(cfg.ticker), mu=mu, sigma=sigma)
    prices.to_csv(DATA_DIR / cfg.pricefile)
