"""
exam_utils.py  —  Exam "glue" functions + high-level pipelines
================================================================
Every function the exam needs that ISN'T an atomic chapter function.
Import with:
    from qrm_lib import exam_utils as eu
    # or
    from qrm_lib.exam_utils import pipeline_copula_portfolio
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Optional, Sequence, Tuple, Union

# --- sibling imports (works both as package and standalone) ---
try:
    from . import chapter1, chapter2, chapter3, chapter4, chapter5
except ImportError:
    import chapter1, chapter2, chapter3, chapter4, chapter5


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING GLUE
# ═══════════════════════════════════════════════════════════════

def load_returns_vector(csv_path: str) -> np.ndarray:
    """Load a single-column returns CSV into a 1-D numpy array."""
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric column found.")
    return df[num_cols[0]].dropna().to_numpy(dtype=float)


def load_returns_matrix(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load multi-asset returns CSV → (array with NaNs kept, column names)."""
    df = pd.read_csv(csv_path)
    num = df.select_dtypes(include=[np.number])
    return num.to_numpy(dtype=float), num.columns.tolist()


def load_prices(csv_path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load a prices CSV, parse dates, sort chronologically."""
    df = pd.read_csv(csv_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    return df


def prices_to_returns(
    prices_df: pd.DataFrame,
    method: str = "discrete",
    date_col: str = "Date",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Prices DataFrame → (returns_matrix, asset_names, last_prices).

    Returns:
        R:           (T-1, n_assets) numpy array of returns
        asset_names: list of column names
        last_prices: 1-D array of most recent prices (for PnL calc)
    """
    rets_df = chapter4.return_calculate(prices_df, date_column=date_col, method=method)
    num = rets_df.select_dtypes(include=[np.number])
    asset_names = num.columns.tolist()
    R = num.to_numpy(dtype=float)

    # last prices from original prices DataFrame
    price_num = prices_df.select_dtypes(include=[np.number])
    last_prices = price_num.iloc[-1].to_numpy(dtype=float)

    return R, asset_names, last_prices


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — SMALL UTILITY GLUE
# ═══════════════════════════════════════════════════════════════

def demean(X: np.ndarray) -> np.ndarray:
    """Column-wise demean, ignoring NaNs."""
    X = np.asarray(X, dtype=float)
    return X - np.nanmean(X, axis=0)


# ── Cov ↔ Corr conversion (Tests 1.2, 1.4, 2.3) ──

def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix → correlation matrix."""
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.diag(cov))
    D_inv = np.diag(1.0 / std)
    corr = D_inv @ cov @ D_inv
    np.fill_diagonal(corr, 1.0)  # fix rounding
    return corr


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Combine correlation matrix + standard deviation vector → covariance matrix.
    This handles Test 2.3: EW Var(λ=0.97) + EW Corr(λ=0.94) → Cov.

    Args:
        corr: (n, n) correlation matrix
        std:  (n,) standard deviation vector (sqrt of variances)
    """
    std = np.asarray(std, dtype=float)
    D = np.diag(std)
    return D @ np.asarray(corr, dtype=float) @ D


def ew_cov_mixed_lambda(
    data: np.ndarray,
    lam_var: float = 0.97,
    lam_corr: float = 0.94,
    do_demean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test 2.3: Covariance using EW Variance (one λ) + EW Correlation (another λ).

    Returns:
        (combined_cov, ew_var_vector, ew_corr_matrix)
    """
    X = np.asarray(data, dtype=float)
    if do_demean:
        X = demean(X)

    # EW covariance with lam_var → extract variances
    cov_var = chapter3.ew_covariance(X, lam=lam_var)
    if isinstance(cov_var, pd.DataFrame):
        cov_var = cov_var.to_numpy(dtype=float)
    ew_var = np.diag(cov_var)       # variance vector
    ew_std = np.sqrt(ew_var)        # std dev vector

    # EW covariance with lam_corr → convert to correlation
    cov_corr = chapter3.ew_covariance(X, lam=lam_corr)
    if isinstance(cov_corr, pd.DataFrame):
        cov_corr = cov_corr.to_numpy(dtype=float)
    ew_corr = cov_to_corr(cov_corr)

    # Combine: D_var @ Corr_corr @ D_var
    combined_cov = corr_to_cov(ew_corr, ew_std)
    return combined_cov, ew_var, ew_corr


# ── VaR / ES for t-distribution (Tests 8.2, 8.3, 8.6) ──

def var_t(nu: float, mu: float, sigma: float, alpha: float = 0.05) -> float:
    """
    Parametric VaR under t-distribution. Returns positive loss.
    Fills gap: standalone t VaR (Test 8.2).
    """
    return -float(stats.t.ppf(alpha, df=nu, loc=mu, scale=sigma))


def var_es_from_simulation(
    sim_data: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    VaR and ES from simulated returns (NOT PnL — just returns).
    Returns positive losses. Fills Tests 8.3, 8.6.
    """
    x = np.asarray(sim_data, dtype=float)
    q = np.quantile(x, alpha)
    var = -float(q)
    tail = x[x <= q]
    es = -float(np.mean(tail)) if tail.size > 0 else var
    return var, es


def simulate_t_var_es(
    nu: float, mu: float, sigma: float,
    alpha: float = 0.05,
    n_sim: int = 100000,
    seed: Optional[int] = None,
) -> Dict:
    """
    Complete Test 8.2/8.3/8.5/8.6: parametric + simulated VaR/ES under t.

    Returns dict with:
        parametric_var, parametric_es,
        simulated_var, simulated_es,
        simulated_data
    """
    # Parametric
    p_var = var_t(nu, mu, sigma, alpha)
    p_es = chapter5.calculate_es_t(nu, mu, sigma, alpha)

    # Simulated
    rng = np.random.default_rng(seed)
    sim = stats.t.rvs(df=nu, loc=mu, scale=sigma, size=n_sim, random_state=rng)
    s_var, s_es = var_es_from_simulation(sim, alpha)

    return {
        "parametric_var": p_var,
        "parametric_es": p_es,
        "simulated_var": s_var,
        "simulated_es": s_es,
        "simulated_data": sim,
    }


# ── Simulate with Higham fix (Test 5.4) ──

def simulate_normal_higham(
    cov_matrix: np.ndarray,
    nsim: int,
    mean: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Multivariate normal simulation using HIGHAM PSD fix + Cholesky.
    Fills Test 5.4 (simulate_normal_cholesky only uses near_psd).
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]
    mu = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)

    # Higham fix
    cov_fix = chapter3.higham_nearest_psd(cov)
    cov_fix = 0.5 * (cov_fix + cov_fix.T)

    # Cholesky
    try:
        L = np.linalg.cholesky(cov_fix)
    except np.linalg.LinAlgError:
        L = chapter3.chol_psd(cov_fix)

    if seed is not None:
        np.random.seed(seed)
    z = np.random.randn(n, nsim)
    out = (L @ z).T + mu
    return out


# ── T-Regression: MLE regression with t-distributed errors (Test 7.3) ──

def mle_regression_t(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """
    MLE regression with t-distributed errors (Test 7.3).

    Model: y = X @ beta + epsilon,  epsilon ~ t(nu, 0, sigma)

    Returns dict with:
        beta, sigma, nu, log_likelihood, aic, aicc, residuals
    """
    from scipy.optimize import minimize as sp_minimize

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    # OLS initial estimates
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    sigma_init = float(np.std(resid_ols))

    def neg_ll(params):
        sigma = params[0]
        nu = params[1]
        beta = params[2:]
        if sigma <= 0 or nu <= 2.0:
            return np.inf
        resid = y - X @ beta
        return -float(np.sum(stats.t.logpdf(resid, df=nu, loc=0, scale=sigma)))

    x0 = np.concatenate(([sigma_init, 5.0], beta_ols))
    bounds = [(1e-6, None), (2.0001, None)] + [(None, None)] * p

    result = sp_minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        # Retry with different initial nu
        x0[1] = 10.0
        result = sp_minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

    sigma_hat = result.x[0]
    nu_hat = result.x[1]
    beta_hat = result.x[2:]
    ll = -result.fun
    n_params = p + 2  # beta (p) + sigma + nu
    aic = 2 * n_params - 2 * ll
    aicc = chapter1.calculate_aicc(ll, n, n_params)
    residuals = y - X @ beta_hat

    return {
        "beta": beta_hat,
        "sigma": float(sigma_hat),
        "nu": float(nu_hat),
        "log_likelihood": float(ll),
        "aic": float(aic),
        "aicc": float(aicc),
        "residuals": residuals,
        "n_params": n_params,
    }


def is_psd(A: np.ndarray, tol: float = -1e-8) -> bool:
    """Check if a matrix is positive semi-definite."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    return float(np.linalg.eigvalsh(A).min()) >= tol


def min_eigenvalue(A: np.ndarray) -> float:
    """Return the smallest eigenvalue of a symmetric matrix."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    return float(np.linalg.eigvalsh(A).min())


def var_es_from_pnl(pnl: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Empirical VaR and ES from a PnL sample.
    Convention: PnL is + = gain, – = loss.
    Returns (VaR, ES) as POSITIVE loss numbers.
    """
    pnl = np.asarray(pnl, dtype=float)
    q = np.quantile(pnl, alpha)
    var = -float(q)
    tail = pnl[pnl <= q]
    es = -float(np.mean(tail)) if tail.size > 0 else var
    return var, es


def portfolio_pnl(
    sim_returns: np.ndarray,
    holdings: np.ndarray,
    current_prices: np.ndarray,
) -> np.ndarray:
    """
    Convert simulated returns → portfolio PnL in dollars.

    Args:
        sim_returns:    (n_sim, n_assets) simulated returns
        holdings:       (n_assets,) number of shares/units
        current_prices: (n_assets,) current price per unit

    Returns:
        1-D array of portfolio PnL (n_sim,)
    """
    h = np.asarray(holdings, dtype=float)
    p = np.asarray(current_prices, dtype=float)
    dollar_pos = h * p                      # (n_assets,) dollar position
    asset_pnl = sim_returns * dollar_pos    # (n_sim, n_assets)
    return asset_pnl.sum(axis=1)            # (n_sim,)


def asset_pnl_matrix(
    sim_returns: np.ndarray,
    holdings: np.ndarray,
    current_prices: np.ndarray,
) -> np.ndarray:
    """
    Like portfolio_pnl but returns the FULL (n_sim, n_assets) PnL matrix.
    Useful when the exam asks for per-asset VaR/ES.
    """
    h = np.asarray(holdings, dtype=float)
    p = np.asarray(current_prices, dtype=float)
    return sim_returns * (h * p)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — EXAM PIPELINES (one per question type)
# ═══════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# Pipeline A: Univariate moments + Normal vs t fit
# ------------------------------------------------------------------
def pipeline_moments_and_fit(
    data: np.ndarray,
) -> Dict:
    """
    Workflow A — moments + distribution selection.

    Args:
        data: 1-D returns array (or pass csv_path and call load_returns_vector first)

    Returns dict with:
        mean, variance, skewness, excess_kurtosis,
        fit (full fit_normal_vs_t_aicc result),
        best_model
    """
    x = np.asarray(data, dtype=float)
    m, v, s, ek = chapter1.first4_moments(x)
    fit = chapter1.fit_normal_vs_t_aicc(x)
    return {
        "mean": m,
        "variance": v,
        "skewness": s,
        "excess_kurtosis": ek,
        "fit": fit,
        "best_model": fit["best_model"],
    }


# ------------------------------------------------------------------
# Pipeline B: Univariate VaR / ES (empirical + parametric)
# ------------------------------------------------------------------
def pipeline_var_es_univariate(
    data: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Workflow B — VaR/ES under empirical, Normal, and t assumptions.

    Returns dict with:
        empirical_var, empirical_es,
        normal_var, normal_es,
        t_var, t_es,
        fit (distribution params)
    """
    x = np.asarray(data, dtype=float)
    fit = chapter1.fit_normal_vs_t_aicc(x)

    # Empirical
    emp_var, emp_es = chapter5.calculate_es(x, alpha=alpha)

    # Normal parametric
    mu_n, sig_n = fit["mu_norm"], fit["sigma_norm"]
    normal_var = chapter4.var_normal(mu_n, sig_n, alpha=alpha)
    normal_es = chapter5.calculate_es_normal(mu_n, sig_n, alpha=alpha)

    # t parametric
    nu, mu_t, sig_t = fit["nu_t"], fit["mu_t"], fit["sigma_t"]
    t_es = chapter5.calculate_es_t(nu, mu_t, sig_t, alpha=alpha)
    t_var = -float(stats.t.ppf(alpha, df=nu, loc=mu_t, scale=sig_t))

    return {
        "empirical_var": emp_var,
        "empirical_es": emp_es,
        "normal_var": normal_var,
        "normal_es": normal_es,
        "t_var": t_var,
        "t_es": t_es,
        "fit": fit,
        "alpha": alpha,
    }


# ------------------------------------------------------------------
# Pipeline C: Exponentially Weighted Covariance
# ------------------------------------------------------------------
def pipeline_ew_covariance(
    data: Union[np.ndarray, pd.DataFrame],
    lam: float = 0.97,
    do_demean: bool = True,
) -> Dict:
    """
    Workflow C — EW covariance (+ optionally correlation).

    Returns dict with:
        ew_cov, ew_corr, lam, is_psd, min_eig
    """
    if isinstance(data, pd.DataFrame):
        X = data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    else:
        X = np.asarray(data, dtype=float)

    if do_demean:
        X = demean(X)

    ew_cov = chapter3.ew_covariance(X, lam=lam)
    if isinstance(ew_cov, pd.DataFrame):
        ew_cov = ew_cov.to_numpy(dtype=float)

    # derive correlation
    std = np.sqrt(np.diag(ew_cov))
    D_inv = np.diag(1.0 / std)
    ew_corr = D_inv @ ew_cov @ D_inv

    return {
        "ew_cov": ew_cov,
        "ew_corr": ew_corr,
        "lam": lam,
        "is_psd": is_psd(ew_cov),
        "min_eig": min_eigenvalue(ew_cov),
    }


# ------------------------------------------------------------------
# Pipeline D: Missing data → pairwise cov → PSD fix → PCA table
# ------------------------------------------------------------------
def pipeline_missing_psd_pca(
    data: np.ndarray,
    fix_method: str = "higham",
    do_demean: bool = True,
) -> Dict:
    """
    Workflow D — pairwise cov with missing data, PSD repair, PCA variance.

    Args:
        data: (n_obs, n_assets) with possible NaNs
        fix_method: "higham" or "near_psd"

    Returns dict with:
        cov_pairwise, min_eig_before, cov_fixed, min_eig_after,
        pca_cumulative (array), pca_table (DataFrame)
    """
    X = np.asarray(data, dtype=float)
    if do_demean:
        X = demean(X)

    # Pairwise covariance (keeps more data)
    cov_pw = chapter3.missing_cov(X, skip_miss=False)
    cov_pw = 0.5 * (cov_pw + cov_pw.T)  # symmetrize
    min_eig_before = min_eigenvalue(cov_pw)

    # Fix
    if fix_method == "higham":
        cov_fix = chapter3.higham_nearest_psd(cov_pw)
    else:
        cov_fix = chapter3.near_psd(cov_pw)
    cov_fix = 0.5 * (cov_fix + cov_fix.T)
    min_eig_after = min_eigenvalue(cov_fix)

    # PCA cumulative variance
    cum = chapter3.pca_cumulative_variance(cov_fix)
    pca_table = pd.DataFrame({
        "k": np.arange(1, len(cum) + 1),
        "cumulative_explained": cum,
    })

    return {
        "cov_pairwise": cov_pw,
        "min_eig_before": min_eig_before,
        "cov_fixed": cov_fix,
        "min_eig_after": min_eig_after,
        "fix_method": fix_method,
        "pca_cumulative": cum,
        "pca_table": pca_table,
    }


# ------------------------------------------------------------------
# Pipeline E: Copula simulation → portfolio VaR/ES
# ------------------------------------------------------------------
def pipeline_copula_portfolio(
    returns: np.ndarray,
    holdings: np.ndarray,
    current_prices: np.ndarray,
    dist_types: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    n_sim: int = 10000,
    method: str = "spearman",
    seed: Optional[int] = None,
    do_demean: bool = True,
) -> Dict:
    """
    Workflow E — full copula simulation → per-asset + portfolio VaR/ES.

    This is the "big question" pipeline that the exam almost always has.

    Args:
        returns:        (T, n_assets) historical returns matrix
        holdings:       (n_assets,) shares/units
        current_prices: (n_assets,) current prices
        dist_types:     per-asset marginal types, e.g. ["t","t","normal"]
                        Default: all "t"
        alpha:          left-tail probability (0.05 = 95% confidence)
        n_sim:          number of Monte Carlo scenarios
        method:         "spearman" or "pearson" for copula correlation
        seed:           random seed for reproducibility
        do_demean:      whether to demean returns before fitting

    Returns dict with:
        simulated_returns, copula_corr, marginals,
        portfolio_pnl, portfolio_var, portfolio_es,
        asset_pnl_matrix, asset_var_es (list of dicts)
    """
    R = np.asarray(returns, dtype=float)
    h = np.asarray(holdings, dtype=float)
    p = np.asarray(current_prices, dtype=float)

    if do_demean:
        R = demean(R)

    # Copula simulation
    sim_R, cop_corr, marginals = chapter5.simulate_copula_mixed(
        data=R,
        dist_types=dist_types,
        n_sim=n_sim,
        method=method,
        seed=seed,
    )

    # PnL calculation
    a_pnl = asset_pnl_matrix(sim_R, h, p)    # (n_sim, n_assets)
    p_pnl = a_pnl.sum(axis=1)                # (n_sim,)

    # Portfolio VaR/ES
    port_var, port_es = var_es_from_pnl(p_pnl, alpha=alpha)

    # Per-asset VaR/ES
    n_assets = R.shape[1]
    asset_var_es = []
    for j in range(n_assets):
        vj, ej = var_es_from_pnl(a_pnl[:, j], alpha=alpha)
        asset_var_es.append({"asset": j, "var": vj, "es": ej})

    return {
        "simulated_returns": sim_R,
        "copula_corr": cop_corr,
        "marginals": marginals,
        "portfolio_pnl": p_pnl,
        "portfolio_var": port_var,
        "portfolio_es": port_es,
        "asset_pnl_matrix": a_pnl,
        "asset_var_es": asset_var_es,
        "alpha": alpha,
        "method": method,
        "n_sim": n_sim,
    }


def pipeline_copula_portfolio_from_prices(
    prices_csv: str,
    holdings: np.ndarray,
    dist_types: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    n_sim: int = 10000,
    method: str = "spearman",
    seed: Optional[int] = None,
    return_method: str = "discrete",
    date_col: str = "Date",
    do_demean: bool = True,
) -> Dict:
    """
    Convenience wrapper: prices CSV → copula → VaR/ES.
    Handles the full data pipeline internally.
    """
    prices_df = load_prices(prices_csv, date_col=date_col)
    R, asset_names, last_prices = prices_to_returns(prices_df, method=return_method, date_col=date_col)

    result = pipeline_copula_portfolio(
        returns=R,
        holdings=np.asarray(holdings, dtype=float),
        current_prices=last_prices,
        dist_types=dist_types,
        alpha=alpha,
        n_sim=n_sim,
        method=method,
        seed=seed,
        do_demean=do_demean,
    )
    result["asset_names"] = asset_names
    result["last_prices"] = last_prices
    return result


# ------------------------------------------------------------------
# Pipeline F: Cholesky vs PCA comparison
# ------------------------------------------------------------------
def pipeline_cholesky_vs_pca(
    cov_matrix: np.ndarray,
    n_sim: int = 10000,
    pct_exp: float = 0.75,
    seed: int = 1234,
) -> Dict:
    """
    Workflow F — Cholesky vs PCA simulation benchmark.

    Returns dict with:
        time_cholesky, time_pca,
        frobenius_cholesky, frobenius_pca,
        sim_cholesky, sim_pca,
        cov_cholesky, cov_pca
    """
    result = chapter3.benchmark_cholesky_vs_pca(
        cov_matrix=cov_matrix,
        n_simulations=n_sim,
        pct_exp=pct_exp,
        seed=seed,
    )
    return {
        "time_cholesky": result["time_cholesky"],
        "time_pca": result["time_pca"],
        "frobenius_cholesky": result["frobenius_cholesky"],
        "frobenius_pca": result["frobenius_pca"],
        "sim_cholesky": result["sim_cholesky"],
        "sim_pca": result["sim_pca"],
        "cov_cholesky": result["cov_cholesky"],
        "cov_pca": result["cov_pca"],
        "pct_exp": pct_exp,
        "n_sim": n_sim,
    }


# ------------------------------------------------------------------
# Pipeline E2: Multi-alpha copula VaR/ES (Test 9.1)
# ------------------------------------------------------------------
def pipeline_copula_multi_alpha(
    returns: np.ndarray,
    holdings: np.ndarray,
    current_prices: np.ndarray,
    alphas: Sequence[float] = (0.05, 0.01),
    dist_types: Optional[Sequence[str]] = None,
    n_sim: int = 10000,
    method: str = "spearman",
    seed: Optional[int] = None,
    do_demean: bool = True,
) -> Dict:
    """
    Copula simulation with VaR/ES reported at MULTIPLE alpha levels.

    Runs the copula simulation ONCE (expensive), then computes VaR/ES
    for each alpha in `alphas` (cheap).

    Returns dict with:
        simulated_returns, copula_corr, marginals,
        portfolio_pnl, asset_pnl_matrix,
        results_by_alpha: { alpha_val: { portfolio_var, portfolio_es, asset_var_es } }
    """
    R = np.asarray(returns, dtype=float)
    h = np.asarray(holdings, dtype=float)
    p = np.asarray(current_prices, dtype=float)

    if do_demean:
        R = demean(R)

    # Single copula simulation
    sim_R, cop_corr, marginals = chapter5.simulate_copula_mixed(
        data=R,
        dist_types=dist_types,
        n_sim=n_sim,
        method=method,
        seed=seed,
    )

    # PnL
    a_pnl = asset_pnl_matrix(sim_R, h, p)
    p_pnl = a_pnl.sum(axis=1)

    # Multi-alpha VaR/ES
    results_by_alpha = {}
    for a in alphas:
        pv, pe = var_es_from_pnl(p_pnl, alpha=a)
        asset_ve = []
        for j in range(R.shape[1]):
            vj, ej = var_es_from_pnl(a_pnl[:, j], alpha=a)
            asset_ve.append({"asset": j, "var": vj, "es": ej})
        results_by_alpha[a] = {
            "portfolio_var": pv,
            "portfolio_es": pe,
            "asset_var_es": asset_ve,
        }

    return {
        "simulated_returns": sim_R,
        "copula_corr": cop_corr,
        "marginals": marginals,
        "portfolio_pnl": p_pnl,
        "asset_pnl_matrix": a_pnl,
        "results_by_alpha": results_by_alpha,
        "alphas": list(alphas),
        "method": method,
        "n_sim": n_sim,
    }


def pipeline_copula_multi_alpha_from_prices(
    prices_csv: str,
    holdings: np.ndarray,
    alphas: Sequence[float] = (0.05, 0.01),
    dist_types: Optional[Sequence[str]] = None,
    n_sim: int = 10000,
    method: str = "spearman",
    seed: Optional[int] = None,
    return_method: str = "discrete",
    date_col: str = "Date",
    do_demean: bool = True,
) -> Dict:
    """Convenience: prices CSV → copula → multi-alpha VaR/ES."""
    prices_df = load_prices(prices_csv, date_col=date_col)
    R, asset_names, last_prices = prices_to_returns(prices_df, method=return_method, date_col=date_col)

    result = pipeline_copula_multi_alpha(
        returns=R,
        holdings=np.asarray(holdings, dtype=float),
        current_prices=last_prices,
        alphas=alphas,
        dist_types=dist_types,
        n_sim=n_sim,
        method=method,
        seed=seed,
        do_demean=do_demean,
    )
    result["asset_names"] = asset_names
    result["last_prices"] = last_prices
    return result


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — REPORTING HELPERS
# ═══════════════════════════════════════════════════════════════

def print_moments(result: Dict) -> None:
    """Pretty-print pipeline_moments_and_fit result."""
    print(f"  Mean:            {result['mean']:.6f}")
    print(f"  Variance:        {result['variance']:.6f}")
    print(f"  Skewness:        {result['skewness']:.4f}")
    print(f"  Excess Kurtosis: {result['excess_kurtosis']:.4f}")
    fit = result["fit"]
    print(f"\n  Normal fit:  mu={fit['mu_norm']:.6f}, sigma={fit['sigma_norm']:.6f}, AICc={fit['aicc_norm']:.2f}")
    print(f"  t fit:       nu={fit['nu_t']:.2f}, mu={fit['mu_t']:.6f}, sigma={fit['sigma_t']:.6f}, AICc={fit['aicc_t']:.2f}")
    print(f"  Best model:  {result['best_model']}")


def print_var_es(result: Dict) -> None:
    """Pretty-print pipeline_var_es_univariate result."""
    a = result["alpha"]
    print(f"  Alpha = {a} ({(1-a)*100:.0f}% confidence)")
    print(f"  Empirical — VaR: {result['empirical_var']:.6f}  ES: {result['empirical_es']:.6f}")
    print(f"  Normal   — VaR: {result['normal_var']:.6f}  ES: {result['normal_es']:.6f}")
    print(f"  t-dist   — VaR: {result['t_var']:.6f}  ES: {result['t_es']:.6f}")


def print_portfolio_risk(result: Dict) -> None:
    """Pretty-print pipeline_copula_portfolio result."""
    print(f"  Method: Gaussian copula ({result['method']} corr), {result['n_sim']} sims")
    print(f"  Alpha = {result['alpha']}")
    print(f"\n  Portfolio VaR: ${result['portfolio_var']:,.2f}")
    print(f"  Portfolio ES:  ${result['portfolio_es']:,.2f}")
    print(f"\n  Copula correlation:\n{result['copula_corr']}")
    print(f"\n  Per-asset VaR/ES:")
    names = result.get("asset_names", None)
    for item in result["asset_var_es"]:
        label = names[item["asset"]] if names else f"Asset {item['asset']}"
        print(f"    {label}: VaR=${item['var']:,.2f}  ES=${item['es']:,.2f}")


def print_multi_alpha_risk(result: Dict) -> None:
    """Pretty-print pipeline_copula_multi_alpha result."""
    print(f"  Method: Gaussian copula ({result['method']} corr), {result['n_sim']} sims")
    names = result.get("asset_names", None)
    for a in result["alphas"]:
        r = result["results_by_alpha"][a]
        print(f"\n  === Alpha = {a} ({(1-a)*100:.0f}% confidence) ===")
        print(f"  Portfolio VaR: ${r['portfolio_var']:,.2f}")
        print(f"  Portfolio ES:  ${r['portfolio_es']:,.2f}")
        for item in r["asset_var_es"]:
            label = names[item["asset"]] if names else f"Asset {item['asset']}"
            print(f"    {label}: VaR=${item['var']:,.2f}  ES=${item['es']:,.2f}")


# ═══════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("=== Pipeline A: Moments + Fit ===")
    x = np.random.standard_t(df=5, size=500)
    res_a = pipeline_moments_and_fit(x)
    print_moments(res_a)

    print("\n=== Pipeline B: VaR/ES ===")
    res_b = pipeline_var_es_univariate(x, alpha=0.05)
    print_var_es(res_b)

    print("\n=== Pipeline C: EW Covariance ===")
    X_multi = np.random.randn(200, 3)
    res_c = pipeline_ew_covariance(X_multi, lam=0.97)
    print(f"  EW Cov shape: {res_c['ew_cov'].shape}, PSD: {res_c['is_psd']}, min eig: {res_c['min_eig']:.6f}")

    print("\n=== Pipeline D: Missing + PSD + PCA ===")
    X_miss = X_multi.copy()
    X_miss[::5, 0] = np.nan   # inject some NaNs
    X_miss[::7, 2] = np.nan
    res_d = pipeline_missing_psd_pca(X_miss, fix_method="higham")
    print(f"  Min eig before: {res_d['min_eig_before']:.6f}")
    print(f"  Min eig after:  {res_d['min_eig_after']:.6f}")
    print(res_d["pca_table"])

    print("\n=== Pipeline E: Copula → Portfolio VaR/ES ===")
    R_hist = np.random.randn(250, 3) * 0.02
    holdings = np.array([100, 200, 150])
    cur_prices = np.array([50.0, 30.0, 80.0])
    res_e = pipeline_copula_portfolio(
        returns=R_hist,
        holdings=holdings,
        current_prices=cur_prices,
        dist_types=["t", "t", "normal"],
        alpha=0.05,
        n_sim=5000,
        method="spearman",
        seed=42,
    )
    print_portfolio_risk(res_e)

    print("\n=== Pipeline F: Cholesky vs PCA ===")
    cov_test = np.cov(X_multi, rowvar=False)
    res_f = pipeline_cholesky_vs_pca(cov_test, n_sim=5000, pct_exp=0.75, seed=42)
    print(f"  Cholesky time: {res_f['time_cholesky']:.4f}s, Frob: {res_f['frobenius_cholesky']:.4f}")
    print(f"  PCA time:      {res_f['time_pca']:.4f}s, Frob: {res_f['frobenius_pca']:.4f}")

    print("\n✅ All pipelines passed.")
