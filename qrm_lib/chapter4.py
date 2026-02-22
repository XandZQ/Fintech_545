import numpy as np
import pandas as pd
import scipy.stats as stats


def return_calculate(prices: pd.DataFrame, date_column: str = "Date", method: str = "discrete") -> pd.DataFrame:
    """
    Calculate arithmetic or log returns from a price DataFrame.

    Args:
        prices: Price table with one date column and one or more price columns.
        date_column: Date column name.
        method: "discrete" or "log" (case-insensitive).

    Returns:
        DataFrame of returns with the same asset columns plus date column.
    """
    if date_column not in prices.columns:
        raise ValueError(f"date_column '{date_column}' not found in DataFrame columns.")
    if len(prices) < 2:
        raise ValueError("prices must contain at least 2 rows.")

    method_norm = method.strip().lower()
    if method_norm not in {"discrete", "log"}:
        raise ValueError("method must be 'discrete' or 'log'.")

    cols = [c for c in prices.columns if c != date_column]
    p = prices[cols].to_numpy(dtype=float)

    ratio = p[1:] / p[:-1]
    if method_norm == "log":
        r = np.log(ratio)
    else:
        r = ratio - 1.0

    out = prices[[date_column]].iloc[1:].reset_index(drop=True).copy()
    out = pd.concat([out, pd.DataFrame(r, columns=cols)], axis=1)
    return out


def var_normal(mu: float, sigma: float, alpha: float = 0.05, n_days: int = 1) -> float:
    """
    Parametric VaR under normal returns.

    Args:
        mu: Mean return per day.
        sigma: Return standard deviation per day.
        alpha: Left-tail probability (0.05 => 95% VaR).
        n_days: Horizon in days.

    Returns:
        Positive VaR loss value in return units.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_days <= 0:
        raise ValueError("n_days must be positive.")

    mu_h = mu * n_days
    sigma_h = sigma * np.sqrt(n_days)
    return -stats.norm.ppf(alpha, loc=mu_h, scale=sigma_h)


def var_delta_normal(
    portfolio_value: float,
    holdings: np.ndarray,
    current_prices: np.ndarray,
    cov_matrix: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Delta-normal VaR in currency units.
    """
    deltas = np.asarray(holdings, dtype=float) * np.asarray(current_prices, dtype=float)
    weights = deltas / float(portfolio_value)
    port_variance = float(weights.T @ cov_matrix @ weights)
    port_sigma = np.sqrt(port_variance)
    return -portfolio_value * stats.norm.ppf(alpha) * port_sigma


def var_historical(
    returns: np.ndarray,
    portfolio_value: float,
    holdings: np.ndarray,
    current_prices: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Historical simulation VaR in currency units.
    """
    rets = np.asarray(returns, dtype=float)
    if rets.ndim == 1:
        rets = rets.reshape(-1, 1)

    prices = np.asarray(current_prices, dtype=float)
    h = np.asarray(holdings, dtype=float)

    sim_prices = prices * (1.0 + rets)
    sim_values = sim_prices @ h
    pnl = sim_values - float(portfolio_value)
    return -np.percentile(pnl, alpha * 100.0)


def var_monte_carlo(
    cov_matrix: np.ndarray,
    portfolio_value: float,
    holdings: np.ndarray,
    current_prices: np.ndarray,
    n_sims: int = 10000,
    alpha: float = 0.05,
) -> float:
    """
    Monte Carlo VaR using multivariate normal returns.
    """
    prices = np.asarray(current_prices, dtype=float)
    h = np.asarray(holdings, dtype=float)
    n_assets = len(prices)

    sim_returns = np.random.multivariate_normal(np.zeros(n_assets), cov_matrix, n_sims)
    sim_prices = prices * (1.0 + sim_returns)
    sim_values = sim_prices @ h
    pnl = sim_values - float(portfolio_value)
    return -np.percentile(pnl, alpha * 100.0)


def expected_shortfall(data: np.ndarray, alpha: float = 0.05) -> float:
    """
    Expected shortfall from a sample of returns/P&L.

    Args:
        data: 1D sample of returns or P&L.
        alpha: Left-tail probability.

    Returns:
        Positive ES loss value.
    """
    x = np.asarray(data, dtype=float)
    var_threshold = np.percentile(x, alpha * 100.0)
    tail = x[x <= var_threshold]
    if tail.size == 0:
        return -var_threshold
    return -float(np.mean(tail))


if __name__ == "__main__":
    print("--- VaR Methods ---")
    prices = np.array([100.0, 50.0, 200.0])
    holdings = np.array([10.0, 20.0, 5.0])
    port_val = np.sum(prices * holdings)

    np.random.seed(42)
    rets = np.random.normal(0, 0.01, (1000, 3))
    rets[:, 1] = 0.5 * rets[:, 0] + 0.5 * np.random.normal(0, 0.01, 1000)
    rets[:, 2] = -0.3 * rets[:, 0] + np.random.normal(0, 0.02, 1000)
    cov_mat = np.cov(rets, rowvar=False)

    print(f"Delta Normal VaR (95%): {var_delta_normal(port_val, holdings, prices, cov_mat, 0.05):.2f}")
    print(f"Historical VaR (95%): {var_historical(rets, port_val, holdings, prices, 0.05):.2f}")
    print(f"Monte Carlo VaR (95%): {var_monte_carlo(cov_mat, port_val, holdings, prices, 5000, 0.05):.2f}")

    sim_prices = prices * (1.0 + rets)
    pnl = sim_prices @ holdings - port_val
    print(f"Historical ES (95%): {expected_shortfall(pnl, 0.05):.2f}")
