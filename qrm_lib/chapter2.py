import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Iterable, Optional, Sequence, Tuple


def calculate_correlations(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson and Spearman correlations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pearson = np.corrcoef(x, y)[0, 1]
    spearman, _ = stats.spearmanr(x, y)
    return float(pearson), float(spearman)


def plot_correlation(x: np.ndarray, y: np.ndarray, title: str = "Correlation Plot") -> None:
    """
    Scatter plot with Pearson and Spearman in title.
    """
    pearson, spearman = calculate_correlations(x, y)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.title(f"{title}\nPearson: {pearson:.2f}, Spearman: {spearman:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def mle_normal(data: np.ndarray) -> Tuple[float, float, float]:
    """
    MLE for Normal distribution.
    """
    x = np.asarray(data, dtype=float)
    mu_hat, sigma_hat = stats.norm.fit(x)
    ll = np.sum(stats.norm.logpdf(x, loc=mu_hat, scale=sigma_hat))
    return float(mu_hat), float(sigma_hat), float(ll)


def mle_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    MLE for linear regression with normal errors.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    def neg_log_likelihood(params: np.ndarray) -> float:
        sigma = params[0]
        beta = params[1:]
        if sigma <= 0:
            return np.inf
        resid = y - X @ beta
        ll = -n / 2 * np.log(2 * np.pi * sigma ** 2) - np.sum(resid ** 2) / (2 * sigma ** 2)
        return -float(ll)

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma_init = np.std(y - X @ beta_ols)
    initial_params = np.concatenate(([sigma_init], beta_ols))

    result = minimize(
        neg_log_likelihood,
        initial_params,
        method="L-BFGS-B",
        bounds=[(1e-6, None)] + [(None, None)] * p,
    )
    sigma_hat = result.x[0]
    beta_hat = result.x[1:]
    ll_val = -result.fun
    return beta_hat, float(sigma_hat), float(ll_val)


def plot_acf_pacf(y: np.ndarray, lags: int = 20, title: str = "Time Series Analysis") -> None:
    """
    Plot time series + ACF + PACF.
    """
    y = np.asarray(y, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(y)
    axes[0].set_title(f"{title} - Time Series")
    sm.graphics.tsa.plot_acf(y, lags=lags, ax=axes[1])
    sm.graphics.tsa.plot_pacf(y, lags=lags, ax=axes[2])
    plt.tight_layout()
    plt.show()


def calculate_aicc(log_likelihood: float, n_obs: int, n_params: int) -> float:
    """
    AICc from log-likelihood.
    """
    if n_obs <= n_params + 1:
        raise ValueError("n_obs must be greater than n_params + 1 for AICc.")
    aic = 2.0 * n_params - 2.0 * log_likelihood
    return float(aic + (2.0 * n_params * (n_params + 1)) / (n_obs - n_params - 1))


def model_result_aicc(model_result, n_obs: Optional[int] = None) -> float:
    """
    AICc from a statsmodels ARIMA result.
    """
    n = int(model_result.nobs if n_obs is None else n_obs)
    k = int(len(model_result.params))
    return float(model_result.aic + (2.0 * k * (k + 1)) / (n - k - 1))


def fit_ar_model(y: np.ndarray, p: int, d: int = 0):
    """
    Fit AR(p) via ARIMA(p,d,0).
    """
    return ARIMA(np.asarray(y, dtype=float), order=(p, d, 0)).fit()


def fit_ma_model(y: np.ndarray, q: int, d: int = 0):
    """
    Fit MA(q) via ARIMA(0,d,q).
    """
    return ARIMA(np.asarray(y, dtype=float), order=(0, d, q)).fit()


def scan_ar_orders_aicc(y: np.ndarray, p_min: int = 1, p_max: int = 10, d: int = 0) -> pd.DataFrame:
    """
    Fit AR orders in [p_min, p_max] and return AIC/AICc/BIC table.
    """
    y = np.asarray(y, dtype=float)
    rows = []
    for p in range(p_min, p_max + 1):
        mdl = fit_ar_model(y, p=p, d=d)
        rows.append(
            {
                "p": p,
                "aic": float(mdl.aic),
                "aicc": model_result_aicc(mdl),
                "bic": float(mdl.bic),
            }
        )
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def select_best_ar_order_aicc(y: np.ndarray, p_min: int = 1, p_max: int = 10, d: int = 0) -> Tuple[int, pd.DataFrame]:
    """
    Select AR order with smallest AICc.
    """
    table = scan_ar_orders_aicc(y, p_min=p_min, p_max=p_max, d=d)
    best_row = table.loc[table["aicc"].idxmin()]
    return int(best_row["p"]), table


def scan_ma_orders_aicc(y: np.ndarray, q_values: Iterable[int], d: int = 0) -> pd.DataFrame:
    """
    Fit specified MA orders and return AIC/AICc/BIC table.
    """
    y = np.asarray(y, dtype=float)
    rows = []
    for q in q_values:
        mdl = fit_ma_model(y, q=int(q), d=d)
        rows.append(
            {
                "q": int(q),
                "aic": float(mdl.aic),
                "aicc": model_result_aicc(mdl),
                "bic": float(mdl.bic),
            }
        )
    return pd.DataFrame(rows).sort_values("q").reset_index(drop=True)


def simulate_ar1(n: int, phi: float, c: float = 0.0, sigma: float = 1.0, burn_in: int = 50) -> np.ndarray:
    """
    Simulate AR(1): y_t = c + phi*y_{t-1} + eps_t.
    """
    y = np.zeros(n + burn_in)
    eps = np.random.normal(0.0, sigma, n + burn_in)
    for t in range(1, n + burn_in):
        y[t] = c + phi * y[t - 1] + eps[t]
    return y[burn_in:]


def simulate_ma1(n: int, theta: float, c: float = 0.0, sigma: float = 1.0, burn_in: int = 50) -> np.ndarray:
    """
    Simulate MA(1): y_t = c + theta*eps_{t-1} + eps_t.
    """
    y = np.zeros(n + burn_in)
    eps = np.random.normal(0.0, sigma, n + burn_in)
    for t in range(1, n + burn_in):
        y[t] = c + theta * eps[t - 1] + eps[t]
    return y[burn_in:]


def simulate_ar_process(
    n: int,
    ar_params: Sequence[float],
    c: float = 0.0,
    sigma: float = 1.0,
    burn_in: int = 100,
) -> np.ndarray:
    """
    Simulate AR(p): y_t = c + sum(phi_i * y_{t-i}) + eps_t.
    """
    phi = np.asarray(ar_params, dtype=float)
    p = len(phi)
    y = np.zeros(n + burn_in + p)
    eps = np.random.normal(0.0, sigma, n + burn_in + p)

    for t in range(p, n + burn_in + p):
        y[t] = c + np.dot(phi, y[t - p : t][::-1]) + eps[t]

    return y[(burn_in + p) :]


def simulate_ma_process(
    n: int,
    ma_params: Sequence[float],
    c: float = 0.0,
    sigma: float = 1.0,
    burn_in: int = 100,
) -> np.ndarray:
    """
    Simulate MA(q): y_t = c + eps_t + sum(theta_i * eps_{t-i}).
    """
    theta = np.asarray(ma_params, dtype=float)
    q = len(theta)
    eps = np.random.normal(0.0, sigma, n + burn_in + q)
    y = np.zeros(n + burn_in + q)

    for t in range(q, n + burn_in + q):
        y[t] = c + eps[t] + np.dot(theta, eps[t - q : t][::-1])

    return y[(burn_in + q) :]


def simulate_ar_orders(
    n: int,
    orders: Iterable[int] = (1, 2, 3),
    phi: float = 0.5,
    c: float = 0.0,
    sigma: float = 1.0,
    burn_in: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Simulate AR(1), AR(2), ... using same coefficient value for each lag.
    """
    out: Dict[str, np.ndarray] = {}
    for p in orders:
        params = [phi] * int(p)
        out[f"AR({p})"] = simulate_ar_process(n=n, ar_params=params, c=c, sigma=sigma, burn_in=burn_in)
    return out


def simulate_ma_orders(
    n: int,
    orders: Iterable[int] = (1, 2, 3),
    theta: float = 0.5,
    c: float = 0.0,
    sigma: float = 1.0,
    burn_in: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Simulate MA(1), MA(2), ... using same coefficient value for each lag.
    """
    out: Dict[str, np.ndarray] = {}
    for q in orders:
        params = [theta] * int(q)
        out[f"MA({q})"] = simulate_ma_process(n=n, ma_params=params, c=c, sigma=sigma, burn_in=burn_in)
    return out


def plot_acf_pacf_grid(series_map: Dict[str, np.ndarray], lags: int = 20, fig_scale: Tuple[int, int] = (14, 4)) -> None:
    """
    Plot ACF/PACF pairs for each named series in a grid.
    """
    names = list(series_map.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(fig_scale[0], fig_scale[1] * n))
    if n == 1:
        axes = np.array([axes])

    for i, name in enumerate(names):
        s = np.asarray(series_map[name], dtype=float)
        sm.graphics.tsa.plot_acf(s, lags=lags, ax=axes[i, 0])
        axes[i, 0].set_title(f"ACF - {name}")
        sm.graphics.tsa.plot_pacf(s, lags=lags, ax=axes[i, 1])
        axes[i, 1].set_title(f"PACF - {name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--- Correlation ---")
    x = np.linspace(-1, 1, 21)
    y = 2 * x
    p, s = calculate_correlations(x, y)
    print(f"Linear (y=2x): Pearson={p:.2f}, Spearman={s:.2f}")

    print("\n--- MLE Normal ---")
    data_norm = np.random.normal(5, 2, 1000)
    mu, sig, _ = mle_normal(data_norm)
    print(f"True(5, 2) -> Est({mu:.2f}, {sig:.2f})")

    print("\n--- AR order by AICc ---")
    y_ar = simulate_ar1(1500, phi=0.6, c=0.2, sigma=1.0)
    best_p, tbl = select_best_ar_order_aicc(y_ar, 1, 6)
    print(tbl)
    print(f"Best AR order: {best_p}")
