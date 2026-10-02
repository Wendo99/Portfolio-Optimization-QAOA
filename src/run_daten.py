"""Data acquisition – portfolio instance.

Stocks as in Baker et al. (2022): GOOG, AMZN, FB (now META), NVDA.
Period: 31.12.2024 close as base → full-year 2025 returns
Result: mu (expected returns) and sigma (covariance matrix),
        frozen in portfolio_daten.npz for all further steps.
"""

from portfolio_qaoa.data import (DataConfig, annualized_moments, load_prices, save,
                                 slide_table)


def main(cfg: DataConfig | None = None) -> None:
    cfg = cfg or DataConfig()
    prices = load_prices(cfg)
    returns = prices.pct_change().dropna()
    print(f"{len(prices)} Handelstage geladen, {len(returns)} Tagesrenditen")

    mu, sigma = annualized_moments(returns, cfg)

    print(slide_table(prices, returns, cfg).map(lambda v: f"{v:.1%}"))
    print("\nKorrelationen:\n", returns.corr().round(2))

    save(cfg, prices, mu, sigma)
    print(f"\nGespeichert: {cfg.datafile}, {cfg.pricefile}")


if __name__ == "__main__":
    main()
