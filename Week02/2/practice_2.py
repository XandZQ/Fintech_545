import numpy as np
import pandas as pd


def ew_cov(x: np.ndarray, lam: float) -> np.ndarray:
    # Exponentially weighted covariance with normalized weights
    n = x.shape[0]
    weights = (1.0 - lam) * np.power(lam, np.arange(n - 1, -1, -1))
    weights = weights / weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    xc = x - mean
    cov = (xc * weights[:, None]).T @ xc
    return cov


def main():
    df = pd.read_csv("test2.csv")
    x = df.to_numpy(dtype=float)

    # 2.1 EW Covariance, lambda=0.97
    cov_097 = ew_cov(x, 0.97)
    pd.DataFrame(cov_097).to_csv("testout_2.1.csv", index=False)

    # 2.2 EW Correlation, lambda=0.94
    cov_094 = ew_cov(x, 0.94)
    sd_inv = 1.0 / np.sqrt(np.diag(cov_094))
    corr_094 = np.diag(sd_inv) @ cov_094 @ np.diag(sd_inv)
    pd.DataFrame(corr_094).to_csv("testout_2.2.csv", index=False)

    # 2.3 EW Cov with EW Var(lambda=0.97) & EW Correlation(lambda=0.94)
    sd1 = np.sqrt(np.diag(cov_097))
    sd_inv = 1.0 / np.sqrt(np.diag(cov_094))
    cout = np.diag(sd1) @ np.diag(sd_inv) @ cov_094 @ np.diag(sd_inv) @ np.diag(sd1)
    pd.DataFrame(cout).to_csv("testout_2.3.csv", index=False)


if __name__ == "__main__":
    main()
