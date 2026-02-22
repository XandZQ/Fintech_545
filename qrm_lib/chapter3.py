import time
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


def chol_psd(a: np.ndarray) -> np.ndarray:
    """
    Cholesky-like factorization for PSD matrices.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros_like(a, dtype=float)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        if 0.0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    return root


def near_psd(a: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Rebonato-Jackel style nearest PSD adjustment.
    """
    out = np.array(a, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("Input matrix must be square.")

    out = 0.5 * (out + out.T)
    invSD = None

    if not np.allclose(np.diag(out), 1.0):
        std = np.sqrt(np.diag(out))
        invSD = np.diag(1.0 / std)
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    t = 1.0 / ((vecs * vecs) @ vals)
    t = np.maximum(t, 0.0)
    T = np.diag(np.sqrt(t))
    L = np.diag(np.sqrt(vals))
    B = T @ vecs @ L
    out = B @ B.T
    out = 0.5 * (out + out.T)

    if invSD is not None:
        std = 1.0 / np.diag(invSD)
        D = np.diag(std)
        out = D @ out @ D
    return out


def _get_aplus(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(a)
    vals = np.diag(np.maximum(vals, 0.0))
    return vecs @ vals @ vecs.T


def _get_ps(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    w05 = np.sqrt(w)
    iw = np.linalg.inv(w05)
    return iw @ _get_aplus(w05 @ a @ w05) @ iw


def _get_pu(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    np.fill_diagonal(out, 1.0)
    return out


def _wgt_norm(a: np.ndarray, w: np.ndarray) -> float:
    w05 = np.sqrt(w)
    x = w05 @ a @ w05
    return float(np.sum(x * x))


def higham_nearest_psd(
    pc: np.ndarray,
    w: Optional[np.ndarray] = None,
    epsilon: float = 1e-9,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Higham nearest PSD algorithm (correlation-constrained projection).
    """
    Yk = np.array(pc, dtype=float, copy=True)
    n = Yk.shape[0]
    if w is None:
        w = np.diag(np.ones(n))

    invSD = None
    if not np.allclose(np.diag(Yk), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD

    deltaS = np.zeros_like(Yk)
    Yo = Yk.copy()
    norml = np.finfo(float).max

    for _ in range(max_iter):
        Rk = Yk - deltaS
        Xk = _get_ps(Rk, w)
        deltaS = Xk - Rk
        Yk = _get_pu(Xk)
        norm = _wgt_norm(Yk - Yo, w)
        min_eig = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and min_eig > -epsilon:
            break
        norml = norm

    Yk = 0.5 * (Yk + Yk.T)

    if invSD is not None:
        D = np.diag(1.0 / np.diag(invSD))
        Yk = D @ Yk @ D
    return Yk


def missing_cov(x: np.ndarray, skip_miss: bool = True) -> np.ndarray:
    """
    Covariance with missing values (np.nan).
    """
    x = np.asarray(x, dtype=float)
    _, m = x.shape

    if skip_miss:
        mask = ~np.isnan(x).any(axis=1)
        return np.cov(x[mask].T)

    out = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            mask = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            if np.sum(mask) > 1:
                c = np.cov(x[mask, i], x[mask, j])[0, 1]
                out[i, j] = c
                if i != j:
                    out[j, i] = c
            else:
                out[i, j] = np.nan
                out[j, i] = np.nan
    return out


def exponential_weights(n: int, lam: float) -> np.ndarray:
    """
    Normalized exponential weights matching week03 convention.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1).")
    w = np.empty(n, dtype=float)
    for i in range(n):
        w[i] = (1.0 - lam) * (lam ** (i + 1))
    w /= np.sum(w)
    return w


def ew_covariance(
    data: Union[pd.DataFrame, np.ndarray],
    lam: float = 0.97,
    date_column: str = "Date",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Exponentially weighted covariance matrix.

    If `data` is DataFrame, keeps DataFrame output with column labels.
    """
    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        df = data.copy()
        if date_column in df.columns:
            df = df.loc[:, df.columns != date_column]
        cols = list(df.columns)
        x = df.to_numpy(dtype=float)
    else:
        x = np.asarray(data, dtype=float)
        cols = None

    n = x.shape[0]
    w = exponential_weights(n, lam)[::-1]
    mu = np.sum(x * w[:, None], axis=0)
    xc = x - mu
    cov = xc.T @ (xc * w[:, None])

    if is_df:
        return pd.DataFrame(cov, index=cols, columns=cols)
    return cov


def pca_cumulative_variance(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Cumulative explained variance from covariance matrix eigenvalues.
    """
    vals = np.linalg.eigvalsh(np.asarray(cov_matrix, dtype=float))[::-1]
    vals = np.maximum(vals, 0.0)
    total = np.sum(vals)
    if total <= 0.0:
        return np.zeros_like(vals)
    return np.cumsum(vals) / total


def simulate_pca(
    a: np.ndarray,
    nsim: int,
    nval: Optional[int] = None,
    pct_exp: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    PCA-based simulation.

    Args:
        a: Covariance matrix.
        nsim: Number of simulated draws.
        nval: Fixed number of factors to keep.
        pct_exp: If provided, choose smallest factor count hitting this variance.
        seed: Optional random seed.
    """
    cov = np.asarray(a, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    total_var_all = float(np.sum(vals))

    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    pos_mask = vals >= 1e-8
    vals = vals[pos_mask]
    vecs = vecs[:, pos_mask]

    if pct_exp is not None:
        if not (0.0 < pct_exp <= 1.0):
            raise ValueError("pct_exp must be in (0, 1].")
        running = np.cumsum(vals) / np.sum(vals)
        keep = int(np.searchsorted(running, pct_exp) + 1)
        vals = vals[:keep]
        vecs = vecs[:, :keep]
    elif nval is not None:
        keep = min(int(nval), len(vals))
        vals = vals[:keep]
        vecs = vecs[:, :keep]

    explained = 0.0 if total_var_all <= 0 else float(np.sum(vals) / total_var_all) * 100.0
    print(f"Simulating with {len(vals)} PC Factors explaining {explained:.2f}% total variance")

    if seed is not None:
        np.random.seed(seed)
    B = vecs @ np.diag(np.sqrt(vals))
    r = np.random.randn(len(vals), nsim)
    return (B @ r).T


def simulate_normal_cholesky(
    cov_matrix: np.ndarray,
    nsim: int,
    mean: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Multivariate normal simulation using Cholesky root, with PSD fixing if needed.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]
    mu = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        fixed = near_psd(cov, epsilon=epsilon)
        try:
            L = np.linalg.cholesky(fixed)
        except np.linalg.LinAlgError:
            L = chol_psd(fixed)

    if seed is not None:
        np.random.seed(seed)
    z = np.random.randn(n, nsim)
    out = (L @ z).T
    out += mu
    return out


def conditional_bivariate_stats(mean: np.ndarray, cov: np.ndarray, x1: float) -> Tuple[float, float]:
    """
    Conditional mean/variance of X2 | X1 = x1 for a bivariate normal.
    """
    m = np.asarray(mean, dtype=float).reshape(-1)
    s = np.asarray(cov, dtype=float)
    mu1, mu2 = m[0], m[1]
    s11, s22 = s[0, 0], s[1, 1]
    s21 = s[1, 0]
    mu_cond = mu2 + s21 / s11 * (x1 - mu1)
    var_cond = s22 - s21 * s21 / s11
    return float(mu_cond), float(var_cond)


def simulate_conditional_bivariate(
    mean: np.ndarray,
    cov: np.ndarray,
    x1: float,
    nsim: int = 10000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate X2 | X1=x1 via Cholesky decomposition.
    """
    m = np.asarray(mean, dtype=float).reshape(-1)
    s = np.asarray(cov, dtype=float)
    L = np.linalg.cholesky(s)
    z1 = (x1 - m[0]) / L[0, 0]
    if seed is not None:
        np.random.seed(seed)
    z2 = np.random.randn(nsim)
    x2 = L[1, 0] * z1 + L[1, 1] * z2 + m[1]
    return x2


def benchmark_cholesky_vs_pca(
    cov_matrix: np.ndarray,
    n_simulations: int = 10000,
    pct_exp: float = 0.75,
    seed: Optional[int] = 1234,
) -> dict:
    """
    Compare Cholesky simulation vs PCA simulation by time and Frobenius norm.
    """
    cov = np.asarray(cov_matrix, dtype=float)

    t0 = time.time()
    sim_chol = simulate_normal_cholesky(cov, n_simulations, seed=seed)
    t_chol = time.time() - t0
    cov_chol = np.cov(sim_chol, rowvar=False)

    t1 = time.time()
    sim_pca = simulate_pca(cov, n_simulations, pct_exp=pct_exp, seed=seed)
    t_pca = time.time() - t1
    cov_pca = np.cov(sim_pca, rowvar=False)

    fro_chol = float(np.linalg.norm(cov_chol - cov, ord="fro"))
    fro_pca = float(np.linalg.norm(cov_pca - cov, ord="fro"))

    return {
        "time_cholesky": t_chol,
        "time_pca": t_pca,
        "cov_cholesky": cov_chol,
        "cov_pca": cov_pca,
        "frobenius_cholesky": fro_chol,
        "frobenius_pca": fro_pca,
        "sim_cholesky": sim_chol,
        "sim_pca": sim_pca,
    }


if __name__ == "__main__":
    print("--- Cholesky PSD ---")
    sigma = np.full((5, 5), 0.9)
    np.fill_diagonal(sigma, 1.0)
    root = chol_psd(sigma)
    print("Reconstructed error:", np.max(np.abs(root @ root.T - sigma)))

    print("\n--- Near PSD ---")
    bad = np.array([[1.0, 1.2], [1.2, 1.0]])
    fixed = near_psd(bad)
    print("Min eigen (fixed):", np.min(np.linalg.eigvalsh(fixed)))

    print("\n--- PCA Simulation ---")
    sim = simulate_pca(sigma, 1000, nval=3)
    print("Simulated Covariance shape:", np.cov(sim.T).shape)
