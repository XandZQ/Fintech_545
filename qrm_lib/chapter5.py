import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.optimize import minimize
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .chapter3 import near_psd
except ImportError:
    from chapter3 import near_psd


def _sample_var_cut(x: np.ndarray, alpha: float) -> float:
    """
    Week05.jl style alpha-quantile estimate:
    average of floor(alpha*n) and ceil(alpha*n) order statistics.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    if n == 0:
        raise ValueError("Input sample must not be empty.")

    k = alpha * n
    iup = int(np.ceil(k))
    idn = int(np.floor(k))
    iup = min(max(iup, 1), n)
    idn = min(max(idn, 1), n)
    return 0.5 * (xs[iup - 1] + xs[idn - 1])


def calculate_es(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Empirical VaR/ES from sample.

    Args:
        data: 1D sample of returns or P&L.
        alpha: Left-tail probability.

    Returns:
        (VaR, ES), both reported as positive losses.
    """
    x = np.asarray(data, dtype=float)
    var_cut = _sample_var_cut(x, alpha)
    tail = x[x <= var_cut]
    return -var_cut, -float(np.mean(tail))


def calculate_es_normal(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """
    Closed-form ES for normal returns, reported as positive loss.
    """
    q = stats.norm.ppf(alpha)
    return -mu + sigma * stats.norm.pdf(q) / alpha


def calculate_es_t(nu: float, mu: float, sigma: float, alpha: float = 0.05) -> float:
    """
    ES for Student-t returns by numerical integration.
    """
    q = stats.t.ppf(alpha, df=nu, loc=mu, scale=sigma)

    def integrand(x: float) -> float:
        return x * stats.t.pdf(x, df=nu, loc=mu, scale=sigma)

    integral, _ = quad(integrand, -np.inf, q)
    return -integral / alpha


def fit_general_t(data: np.ndarray) -> Tuple[float, float, float]:
    """
    MLE fit for t location-scale parameters (mu, sigma, nu).
    """
    x = np.asarray(data, dtype=float)

    def neg_log_likelihood(params: np.ndarray) -> float:
        mu, sigma, nu = params
        if sigma <= 0.0 or nu <= 2.0:
            return np.inf
        return -np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=sigma))

    mu_init = float(np.mean(x))
    var = float(np.var(x))
    ex_kurt = float(stats.kurtosis(x, fisher=True, bias=False))
    if ex_kurt > 1e-8:
        nu_init = max(2.1, 6.0 / ex_kurt + 4.0)
    else:
        nu_init = 30.0
    sigma_init = np.sqrt(max(var * (nu_init - 2.0) / nu_init, 1e-12))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([mu_init, sigma_init, nu_init], dtype=float),
        bounds=[(None, None), (1e-6, None), (2.0001, None)],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"fit_general_t optimization failed: {result.message}")

    mu, sigma, nu = result.x
    return float(mu), float(sigma), float(nu)


def _normalize_dist_type(label: str) -> str:
    t = str(label).strip().lower()
    if t in {"t", "student-t", "student_t", "studentt"}:
        return "t"
    if t in {"normal", "gaussian", "norm"}:
        return "normal"
    raise ValueError(f"Unsupported dist type '{label}'. Use 't' or 'normal'.")


def fit_copula_marginals(
    data: np.ndarray,
    dist_types: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Fit marginal distributions and transform observations to U(0,1).

    Args:
        data: Historical sample (n_obs, n_assets).
        dist_types: per-asset dist labels ('t' or 'normal').
            If None, defaults to all 't' (course week05 style).
        eps: clipping level to avoid exact 0/1.

    Returns:
        u_data: transformed uniforms, shape (n_obs, n_assets)
        marginals: list of dicts in asset-column order.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be a 2D array.")
    _, m = x.shape

    if dist_types is None:
        dist_types = ["t"] * m
    if len(dist_types) != m:
        raise ValueError("dist_types length must match number of columns in data.")

    u_data = np.zeros_like(x)
    marginals: List[Dict[str, float]] = []

    for i in range(m):
        dist = _normalize_dist_type(dist_types[i])
        xi = x[:, i]
        if dist == "normal":
            mu, sigma = stats.norm.fit(xi)
            u = stats.norm.cdf(xi, loc=mu, scale=sigma)
            marginals.append({"dist": "normal", "mu": float(mu), "sigma": float(sigma)})
        else:
            mu, sigma, nu = fit_general_t(xi)
            u = stats.t.cdf(xi, df=nu, loc=mu, scale=sigma)
            marginals.append({"dist": "t", "mu": float(mu), "sigma": float(sigma), "nu": float(nu)})
        u_data[:, i] = np.clip(u, eps, 1.0 - eps)

    return u_data, marginals


def fit_gaussian_copula_corr(
    u_data: np.ndarray,
    method: str = "pearson",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Fit Gaussian copula correlation matrix from transformed uniform data.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    z_data = stats.norm.ppf(np.asarray(u_data, dtype=float))
    if method == "spearman":
        ranks = np.apply_along_axis(stats.rankdata, 0, z_data)
        corr = np.corrcoef(ranks, rowvar=False)
    else:
        corr = np.corrcoef(z_data, rowvar=False)

    corr = np.asarray(corr, dtype=float)
    if corr.ndim == 0:
        corr = np.array([[1.0]], dtype=float)
    corr = near_psd(corr, epsilon=epsilon)
    return corr


def simulate_copula_from_fitted(
    marginals: List[Dict[str, float]],
    corr: np.ndarray,
    n_sim: int = 10000,
    seed: Optional[int] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Simulate returns from fitted marginals and a Gaussian-copula correlation.
    """
    m = len(marginals)
    if seed is None:
        z_sim = np.random.multivariate_normal(np.zeros(m), corr, n_sim)
    else:
        rng = np.random.default_rng(seed)
        z_sim = rng.multivariate_normal(np.zeros(m), corr, n_sim)
    u_sim = np.clip(stats.norm.cdf(z_sim), eps, 1.0 - eps)

    simulated = np.zeros((n_sim, m))
    for i, info in enumerate(marginals):
        if info["dist"] == "normal":
            simulated[:, i] = stats.norm.ppf(u_sim[:, i], loc=info["mu"], scale=info["sigma"])
        else:
            simulated[:, i] = stats.t.ppf(u_sim[:, i], df=info["nu"], loc=info["mu"], scale=info["sigma"])
    return simulated


def simulate_copula_mixed(
    data: np.ndarray,
    dist_types: Optional[Sequence[str]] = None,
    n_sim: int = 10000,
    method: str = "pearson",
    seed: Optional[int] = None,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    High-level mixed-marginal Gaussian copula simulation.

    Returns:
        simulated_data, copula_corr, marginals
    """
    u_data, marginals = fit_copula_marginals(data, dist_types=dist_types, eps=epsilon)
    corr = fit_gaussian_copula_corr(u_data, method=method, epsilon=epsilon)
    simulated = simulate_copula_from_fitted(marginals, corr, n_sim=n_sim, seed=seed, eps=epsilon)
    return simulated, corr, marginals


def simulate_copula(data: np.ndarray, n_sim: int = 10000, method: str = "pearson") -> np.ndarray:
    """
    Backward-compatible wrapper: Gaussian copula simulation with all t marginals.
    """
    simulated, _, _ = simulate_copula_mixed(
        data=data,
        dist_types=None,
        n_sim=n_sim,
        method=method,
        seed=None,
        epsilon=1e-12,
    )
    return simulated


if __name__ == "__main__":
    print("--- ES Calculation ---")
    data = np.random.normal(0, 1, 10000)
    var, es = calculate_es(data)
    es_theory = calculate_es_normal(0, 1)
    print(f"Empirical ES: {es:.4f}, Theoretical Normal ES: {es_theory:.4f}")

    print("\n--- Fit General T ---")
    t_data = stats.t.rvs(df=5, loc=0, scale=1, size=1000)
    mu_hat, sig_hat, nu_hat = fit_general_t(t_data)
    print(f"True(0, 1, 5) -> Est({mu_hat:.2f}, {sig_hat:.2f}, {nu_hat:.2f})")

    print("\n--- Copula Simulation ---")
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    xy = np.random.multivariate_normal([0.0, 0.0], cov, 500)
    sim_xy, corr_xy, _ = simulate_copula_mixed(xy, dist_types=["normal", "normal"], n_sim=1000, method="pearson", seed=42)
    print("Copula corr used:\n", corr_xy)
    print("Simulated correlation:\n", np.corrcoef(sim_xy, rowvar=False))
