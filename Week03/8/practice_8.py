import numpy as np
import pandas as pd
from scipy.stats import norm, t


ALPHA = 0.05
N_SIM = 10000


def read_first_column(csv_file):
    df = pd.read_csv(csv_file)
    return df.iloc[:, 0].to_numpy(dtype=float)


def empirical_var(x, alpha=ALPHA):
    # VaR is reported as a positive loss number
    q = np.quantile(x, alpha)
    return -q


def save_var_output(file_name, var_abs, var_diff):
    out = pd.DataFrame(
        {
            "VaR Absolute": [var_abs],
            "VaR Diff from Mean": [var_diff],
        }
    )
    out.to_csv(file_name, index=False)


def test_81_var_normal():
    x = read_first_column("test7_1.csv")
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)

    var_absolute = -norm.ppf(ALPHA, loc=mu, scale=sigma)
    var_diff_from_mean = -norm.ppf(ALPHA, loc=0.0, scale=sigma)
    save_var_output("testout8_1.csv", var_absolute, var_diff_from_mean)


def test_82_var_t():
    x = read_first_column("test7_2.csv")
    nu, mu, sigma = t.fit(x)

    var_absolute = -t.ppf(ALPHA, df=nu, loc=mu, scale=sigma)
    var_diff_from_mean = -t.ppf(ALPHA, df=nu, loc=0.0, scale=sigma)
    save_var_output("testout8_2.csv", var_absolute, var_diff_from_mean)


def test_83_var_simulation():
    x = read_first_column("test7_2.csv")
    nu, mu, sigma = t.fit(x)

    sim = t.rvs(df=nu, loc=mu, scale=sigma, size=N_SIM)
    var_absolute = empirical_var(sim)
    var_diff_from_mean = empirical_var(sim - np.mean(sim))
    save_var_output("testout8_3.csv", var_absolute, var_diff_from_mean)


def main():
    # 8.1 = VaR from Normal Distribution
    test_81_var_normal()
    # 8.2 = VaR from T Distribution
    test_82_var_t()
    # 8.3 = VaR from Simulation
    test_83_var_simulation()


if __name__ == "__main__":
    main()
