import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def first4_moments(sample: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate first 4 sample moments: mean, variance, skewness, excess kurtosis.

    Args:
        sample: 1D numeric sample.

    Returns:
        (mean, variance_unbiased, skewness, excess_kurtosis)
    """
    x = np.asarray(sample, dtype=float)
    n = len(x)

    mu_hat = np.mean(x)
    centered = x - mu_hat
    cm2 = np.dot(centered, centered) / n
    sigma2_hat = np.var(x, ddof=1)
    skew_hat = np.sum(centered ** 3) / n / np.sqrt(cm2 ** 3)
    kurt_hat = np.sum(centered ** 4) / n / (cm2 ** 2)
    excess_kurt_hat = kurt_hat - 3.0
    return float(mu_hat), float(sigma2_hat), float(skew_hat), float(excess_kurt_hat)


def calculate_aicc(log_likelihood: float, n_obs: int, n_params: int) -> float:
    """
    Small-sample corrected AIC.

    Args:
        log_likelihood: Model log-likelihood.
        n_obs: Number of observations.
        n_params: Number of fitted parameters.

    Returns:
        AICc value.
    """
    if n_obs <= n_params + 1:
        raise ValueError("n_obs must be greater than n_params + 1 for AICc.")
    aic = 2.0 * n_params - 2.0 * log_likelihood
    return float(aic + (2.0 * n_params * (n_params + 1)) / (n_obs - n_params - 1))


def fit_normal_vs_t_aicc(sample: np.ndarray) -> Dict[str, float]:
    """
    Fit Normal and Student-t distributions and compare by AICc.

    Args:
        sample: 1D numeric sample.

    Returns:
        Dict with fitted parameters, AICc values, and best model label.
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 5:
        raise ValueError("sample size must be >= 5")

    mu_norm, sigma_norm = stats.norm.fit(x)
    ll_norm = np.sum(stats.norm.logpdf(x, loc=mu_norm, scale=sigma_norm))
    aicc_norm = calculate_aicc(ll_norm, n, 2)

    nu_t, mu_t, sigma_t = stats.t.fit(x)
    ll_t = np.sum(stats.t.logpdf(x, df=nu_t, loc=mu_t, scale=sigma_t))
    aicc_t = calculate_aicc(ll_t, n, 3)

    best = "normal" if aicc_norm <= aicc_t else "t"
    return {
        "mu_norm": float(mu_norm),
        "sigma_norm": float(sigma_norm),
        "aicc_norm": float(aicc_norm),
        "nu_t": float(nu_t),
        "mu_t": float(mu_t),
        "sigma_t": float(sigma_t),
        "aicc_t": float(aicc_t),
        "best_model": best,
    }


def plot_pdf_cdf_normal(mu: float = 0.0, sigma: float = 1.0, show_plot: bool = True) -> pd.DataFrame:
    """
    Plot PDF/CDF of Normal(mu, sigma).
    """
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
    df = pd.DataFrame({"x": x, "pdf": pdf, "cdf": cdf})

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(df["x"], df["pdf"], label="PDF")
        axes[0].set_title(f"Normal PDF (mu={mu}, sigma={sigma})")
        axes[0].legend()
        axes[1].plot(df["x"], df["cdf"], label="CDF", color="orange")
        axes[1].set_title(f"Normal CDF (mu={mu}, sigma={sigma})")
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    return df


def test_kurtosis_normal(n_samples: int = 1000, sample_size: int = 10) -> float:
    """
    Monte Carlo test of excess-kurtosis bias for normal samples.
    """
    kurts = np.empty(n_samples)
    for i in range(n_samples):
        sample = np.random.normal(0.0, 1.0, sample_size)
        _, _, _, k = first4_moments(sample)
        kurts[i] = k
    _, p_value = stats.ttest_1samp(kurts, 0.0)
    return float(p_value)


if __name__ == "__main__":
    print("--- Normal Distribution Analysis ---")
    df = plot_pdf_cdf_normal()
    print(f"First 5 rows of PDF/CDF:\n{df.head()}")

    print("\n--- Moments Calculation ---")
    sim = np.random.normal(0, 1, 1000)
    m, s2, sk, k = first4_moments(sim)
    print(f"Mean: {m:.4f}")
    print(f"Variance: {s2:.4f}")
    print(f"Skewness: {sk:.4f}")
    print(f"Excess Kurtosis: {k:.4f}")

    print("\n--- Normal vs t by AICc ---")
    fit = fit_normal_vs_t_aicc(sim)
    print(f"AICc Normal: {fit['aicc_norm']:.4f}")
    print(f"AICc t: {fit['aicc_t']:.4f}")
    print(f"Best: {fit['best_model']}")
