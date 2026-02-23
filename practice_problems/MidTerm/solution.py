from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

# Make project root importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qrm_lib.chapter1 import first4_moments, fit_normal_vs_t_aicc
from qrm_lib.chapter3 import ew_covariance, missing_cov, higham_nearest_psd, near_psd
from qrm_lib.chapter4 import return_calculate, var_normal
from qrm_lib.chapter5 import fit_general_t, calculate_es_normal, calculate_es_t


BASE_DIR = Path(__file__).resolve().parent
ALPHA = 0.05
N_SIM = 200_000


def classify_definiteness(mat: np.ndarray, tol: float = 1e-10) -> str:
    eigvals = np.linalg.eigvalsh(mat)
    min_eig = float(np.min(eigvals))
    if min_eig > tol:
        return "Positive Definite"
    if min_eig >= -tol:
        return "Positive Semi-Definite"
    return "Non Definite"


def var_es_from_pnl(pnl: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    cut = float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= cut]
    return -cut, -float(np.mean(tail))


def simulate_gaussian_copula_with_corr(
    data: np.ndarray,
    n_sim: int = N_SIM,
    method: str = "spearman",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float]]]:
    """
    Reuses qrm_lib fit_general_t + near_psd and follows chapter5.simulate_copula logic,
    but also returns the copula correlation matrix used in simulation.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be a 2D array")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    _, n_assets = x.shape
    eps = 1e-12

    marginals: list[tuple[float, float, float]] = []
    u_data = np.zeros_like(x)
    for i in range(n_assets):
        mu, sigma, nu = fit_general_t(x[:, i])
        marginals.append((mu, sigma, nu))
        u = stats.t.cdf(x[:, i], df=nu, loc=mu, scale=sigma)
        u_data[:, i] = np.clip(u, eps, 1.0 - eps)

    z_data = stats.norm.ppf(u_data)
    if method == "spearman":
        corr = stats.spearmanr(z_data).correlation
    else:
        corr = np.corrcoef(z_data, rowvar=False)

    corr = near_psd(corr, epsilon=1e-12)

    rng = np.random.default_rng(seed)
    z_sim = rng.multivariate_normal(np.zeros(n_assets), corr, size=n_sim)
    u_sim = np.clip(stats.norm.cdf(z_sim), eps, 1.0 - eps)

    sim_rets = np.zeros((n_sim, n_assets))
    for i, (mu, sigma, nu) in enumerate(marginals):
        sim_rets[:, i] = stats.t.ppf(u_sim[:, i], df=nu, loc=mu, scale=sigma)

    return sim_rets, corr, marginals


def solve_problem_1() -> str:
    return (
        "Forecasting focuses on predicting the conditional mean (best point estimate), "
        "while risk modeling focuses on the full tail-aware distribution of outcomes "
        "(variance, skewness, kurtosis, and extreme-loss behavior)."
    )


def solve_problem_2() -> dict:
    x = pd.read_csv(BASE_DIR / "problem2.csv")["X"].to_numpy(dtype=float)

    mean_x, var_x, skew_x, exkurt_x = first4_moments(x)
    fit = fit_normal_vs_t_aicc(x)

    choice_b = "t-distribution" if exkurt_x > 0 else "normal distribution"

    return {
        "data": x,
        "mean": mean_x,
        "variance": var_x,
        "skewness": skew_x,
        "excess_kurtosis": exkurt_x,
        "choice_b": choice_b,
        "fit": fit,
    }


def solve_problem_3(fit: dict) -> dict:
    var_normal_5 = var_normal(fit["mu_norm"], fit["sigma_norm"], alpha=ALPHA)
    var_t_5 = -stats.t.ppf(ALPHA, df=fit["nu_t"], loc=fit["mu_t"], scale=fit["sigma_t"])

    es_normal_5 = calculate_es_normal(fit["mu_norm"], fit["sigma_norm"], alpha=ALPHA)
    es_t_5 = calculate_es_t(fit["nu_t"], fit["mu_t"], fit["sigma_t"], alpha=ALPHA)

    return {
        "var_normal": float(var_normal_5),
        "var_t": float(var_t_5),
        "es_normal": float(es_normal_5),
        "es_t": float(es_t_5),
    }


def solve_problem_4() -> dict:
    data4 = pd.read_csv(BASE_DIR / "problem4.csv")

    cov_lam94 = np.asarray(ew_covariance(data4, lam=0.94), dtype=float)
    std_lam94 = np.sqrt(np.diag(cov_lam94))
    corr_lam94 = cov_lam94 / np.outer(std_lam94, std_lam94)

    cov_lam97 = np.asarray(ew_covariance(data4, lam=0.97), dtype=float)
    vars_lam97 = np.diag(cov_lam97)

    std_lam97 = np.sqrt(vars_lam97)
    mixed_cov = np.outer(std_lam97, std_lam97) * corr_lam94

    explanation = (
        "Use different lambdas when volatility and correlation update at different speeds. "
        "Lower lambda (0.94) adapts faster for correlation; higher lambda (0.97) is smoother for variances."
    )

    return {
        "corr_lam94": corr_lam94,
        "vars_lam97": vars_lam97,
        "mixed_cov": mixed_cov,
        "explanation": explanation,
    }


def solve_problem_5() -> dict:
    data5 = pd.read_csv(BASE_DIR / "problem5.csv").to_numpy(dtype=float)

    pair_cov = missing_cov(data5, skip_miss=False)
    definiteness = classify_definiteness(pair_cov)
    min_eig = float(np.min(np.linalg.eigvalsh(pair_cov)))

    fixed_cov = higham_nearest_psd(pair_cov)

    eigvals = np.linalg.eigvalsh(fixed_cov)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    explained = eigvals / np.sum(eigvals)
    cumulative = np.cumsum(explained)

    pca_table = pd.DataFrame(
        {
            "PC": np.arange(1, len(explained) + 1),
            "Explained": explained,
            "Cumulative": cumulative,
        }
    )

    return {
        "pair_cov": pair_cov,
        "definiteness": definiteness,
        "min_eig": min_eig,
        "fixed_cov": fixed_cov,
        "pca_table": pca_table,
    }


def solve_problem_6() -> dict:
    prices = pd.read_csv(BASE_DIR / "problem6.csv")
    returns = return_calculate(prices, date_column="Date", method="discrete")

    assets = [c for c in returns.columns if c != "Date"]

    returns_dm = returns.copy()
    returns_dm[assets] = returns_dm[assets] - returns_dm[assets].mean(axis=0)

    fit_rows = []
    for a in assets:
        mu, sigma, nu = fit_general_t(returns_dm[a].to_numpy(dtype=float))
        fit_rows.append({"Stock": a, "mu": mu, "sigma": sigma, "nu": nu})
    fit_table = pd.DataFrame(fit_rows)

    sim_rets, copula_corr, _ = simulate_gaussian_copula_with_corr(
        returns_dm[assets].to_numpy(dtype=float),
        n_sim=N_SIM,
        method="spearman",
        seed=42,
    )

    current_prices = prices[assets].iloc[-1].to_numpy(dtype=float)
    holdings = np.full(len(assets), 100.0)
    current_values = current_prices * holdings

    sim_prices = current_prices[None, :] * (1.0 + sim_rets)
    sim_values = sim_prices * holdings[None, :]
    pnl_by_stock = sim_values - current_values[None, :]

    risk_rows = []
    for i, a in enumerate(assets):
        var_i, es_i = var_es_from_pnl(pnl_by_stock[:, i], alpha=ALPHA)
        risk_rows.append({"Stock": a, "VaR_5pct_$": var_i, "ES_5pct_$": es_i})

    total_pnl = np.sum(pnl_by_stock, axis=1)
    var_total, es_total = var_es_from_pnl(total_pnl, alpha=ALPHA)
    risk_rows.append({"Stock": "Total", "VaR_5pct_$": var_total, "ES_5pct_$": es_total})

    risk_table = pd.DataFrame(risk_rows)

    return {
        "fit_table": fit_table,
        "copula_corr": copula_corr,
        "risk_table": risk_table,
    }


def main() -> None:
    print("Problem 1")
    print(solve_problem_1())
    print()

    print("Problem 2")
    p2 = solve_problem_2()
    print(f"Mean: {p2['mean']:.12f}")
    print(f"Variance: {p2['variance']:.12f}")
    print(f"Skewness: {p2['skewness']:.12f}")
    print(f"Excess Kurtosis: {p2['excess_kurtosis']:.12f}")
    print(f"Part b choice (from moments): {p2['choice_b']}")
    print(f"AICc Normal: {p2['fit']['aicc_norm']:.6f}")
    print(f"AICc t: {p2['fit']['aicc_t']:.6f}")
    print(f"Best model by AICc: {p2['fit']['best_model']}")
    print()

    print("Problem 3")
    p3 = solve_problem_3(p2["fit"])
    print(f"VaR 5% (Normal, distance from 0): {p3['var_normal']:.6f}")
    print(f"VaR 5% (t, distance from 0): {p3['var_t']:.6f}")
    print(f"ES 5% (Normal, distance from 0): {p3['es_normal']:.6f}")
    print(f"ES 5% (t, distance from 0): {p3['es_t']:.6f}")
    print()

    print("Problem 4")
    p4 = solve_problem_4()
    print("EW Correlation (lambda=0.94):")
    print(np.round(p4["corr_lam94"], 6))
    print("EW Variances (lambda=0.97):")
    print(np.round(p4["vars_lam97"], 12))
    print("Mixed Covariance Matrix:")
    print(np.round(p4["mixed_cov"], 6))
    print("Why in practice:")
    print(p4["explanation"])
    print()

    print("Problem 5")
    p5 = solve_problem_5()
    print("Pairwise covariance (with missing values):")
    print(np.round(p5["pair_cov"], 6))
    print(f"Min eigenvalue: {p5['min_eig']:.12f}")
    print(f"Definiteness: {p5['definiteness']}")
    print("Higham-fixed covariance:")
    print(np.round(p5["fixed_cov"], 6))
    print("PCA explained and cumulative variance:")
    print(p5["pca_table"].round(6).to_string(index=False))
    print()

    print("Problem 6")
    p6 = solve_problem_6()
    print("Student-t fit per stock (demeaned arithmetic returns):")
    print(p6["fit_table"].round(6).to_string(index=False))
    print("Copula correlation matrix used:")
    print(np.round(p6["copula_corr"], 6))
    print("VaR/ES in dollars (5% alpha):")
    print(p6["risk_table"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
