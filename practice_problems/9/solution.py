from __future__ import annotations

from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Make project root importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qrm_lib.chapter5 import calculate_es, simulate_copula_mixed


ALPHAS = (0.05, 0.01)  # 95% and 99% levels
N_SIM_DEFAULT = 200_000


def _portfolio_pnl_from_returns(sim_returns: np.ndarray, portfolio: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-asset and total PnL from simulated returns.
    """
    prices = portfolio["Starting Price"].to_numpy(dtype=float)
    holdings = portfolio["Holding"].to_numpy(dtype=float)
    current_values = prices * holdings

    sim_prices = prices[None, :] * (1.0 + sim_returns)
    sim_values = sim_prices * holdings[None, :]
    pnl_assets = sim_values - current_values[None, :]
    pnl_total = np.sum(pnl_assets, axis=1)
    return pnl_assets, pnl_total


def _build_risk_table(pnl_assets: np.ndarray, pnl_total: np.ndarray, assets: List[str], alphas=ALPHAS) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(assets):
        row = {"Level": "Asset", "Name": a}
        for alpha in alphas:
            var_a, es_a = calculate_es(pnl_assets[:, i], alpha=alpha)
            pct = int((1 - alpha) * 100)
            row[f"VaR_{pct}_$"] = var_a
            row[f"ES_{pct}_$"] = es_a
        rows.append(row)

    total_row = {"Level": "Portfolio", "Name": "Total"}
    for alpha in alphas:
        var_t, es_t = calculate_es(pnl_total, alpha=alpha)
        pct = int((1 - alpha) * 100)
        total_row[f"VaR_{pct}_$"] = var_t
        total_row[f"ES_{pct}_$"] = es_t
    rows.append(total_row)

    return pd.DataFrame(rows)


def solve_case(
    portfolio_csv: Path,
    returns_csv: Path,
    n_sim: int = N_SIM_DEFAULT,
    corr_method: str = "spearman",
    seed: int = 42,
) -> dict:
    portfolio = pd.read_csv(portfolio_csv)
    returns = pd.read_csv(returns_csv)

    assets = portfolio["Stock"].tolist()
    missing = [a for a in assets if a not in returns.columns]
    if missing:
        raise ValueError(f"These assets are missing in returns file: {missing}")

    returns = returns[assets].copy()
    dist_types = portfolio["Distribution"].astype(str).tolist()
    sim_returns, corr, marginals = simulate_copula_mixed(
        data=returns.to_numpy(dtype=float),
        dist_types=dist_types,
        n_sim=n_sim,
        method=corr_method,
        seed=seed,
    )
    fit_info = {asset: marginals[i] for i, asset in enumerate(assets)}
    pnl_assets, pnl_total = _portfolio_pnl_from_returns(sim_returns, portfolio)
    risk_table = _build_risk_table(pnl_assets, pnl_total, assets=assets, alphas=ALPHAS)

    return {
        "portfolio": portfolio,
        "fit_info": fit_info,
        "copula_corr": corr,
        "risk_table": risk_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VaR/ES on two confidence levels from copula-simulated values."
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default="test9_1_portfolio.csv",
        help="Portfolio CSV filename (relative to this script folder by default).",
    )
    parser.add_argument(
        "--returns",
        type=str,
        default="test9_1_returns.csv",
        help="Returns CSV filename (relative to this script folder by default).",
    )
    parser.add_argument("--n-sim", type=int, default=N_SIM_DEFAULT, help="Number of copula simulations.")
    parser.add_argument(
        "--corr-method",
        type=str,
        default="spearman",
        choices=["spearman", "pearson"],
        help="Correlation estimator for copula fit.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    portfolio_csv = (base / args.portfolio).resolve()
    returns_csv = (base / args.returns).resolve()

    out = solve_case(
        portfolio_csv=portfolio_csv,
        returns_csv=returns_csv,
        n_sim=args.n_sim,
        corr_method=args.corr_method,
        seed=args.seed,
    )

    print("Copula correlation matrix used:")
    print(np.round(out["copula_corr"], 6))
    print("\nMarginal fits:")
    for asset, info in out["fit_info"].items():
        print(asset, info)

    print("\nVaR/ES from simulated values (2 levels: 95%, 99%):")
    print(out["risk_table"].round(3).to_string(index=False))

    out_csv = base / "test9_1_solution_output.csv"
    out["risk_table"].to_csv(out_csv, index=False)
    print(f"\nSaved risk table to: {out_csv}")


if __name__ == "__main__":
    main()
