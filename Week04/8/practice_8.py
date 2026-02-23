import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t
from scipy.integrate import quad

ALPHA = 0.05
N_SIM = 10000


def read_first_column(csv_file):
    df = pd.read_csv(csv_file)
    return df.iloc[:, 0].to_numpy(dtype=float)


def empirical_var(x, alpha=ALPHA):
    # VaR is reported as a positive loss number
    q = np.quantile(x, alpha)
    return -q


def empirical_es(x, alpha=ALPHA):
    # ES is average loss in the worst alpha tail
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return -np.mean(tail)


def save_var_output(file_name, var_abs, var_diff):
    out = pd.DataFrame(
        {
            "VaR Absolute": [var_abs],
            "VaR Diff from Mean": [var_diff],
        }
    )
    out.to_csv(file_name, index=False)


def save_es_output(file_name, es_abs, es_diff):
    out = pd.DataFrame(
        {
            "ES Absolute": [es_abs],
            "ES Diff from Mean": [es_diff],
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

def calculate_es_normal(mu: float, sigma: float, alpha: float = ALPHA) -> float:
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


def test_84_es_normal():
    x = read_first_column("test7_1.csv")
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)

    es_absolute = calculate_es_normal(mu, sigma, ALPHA)
    es_diff_from_mean = calculate_es_normal(0.0, sigma, ALPHA)
    save_es_output("testout8_4.csv", es_absolute, es_diff_from_mean)


def test_85_es_t():
    x = read_first_column("test7_2.csv")
    nu, mu, sigma = t.fit(x)

    es_absolute = calculate_es_t(nu, mu, sigma, ALPHA)
    es_diff_from_mean = calculate_es_t(nu, 0.0, sigma, ALPHA)
    save_es_output("testout8_5.csv", es_absolute, es_diff_from_mean)
    return es_absolute, es_diff_from_mean


def test_86_es_simulation():
    x = read_first_column("test7_2.csv")
    nu, mu, sigma = t.fit(x)

    sim = t.rvs(df=nu, loc=mu, scale=sigma, size=N_SIM)
    es_absolute = empirical_es(sim)
    es_diff_from_mean = empirical_es(sim - np.mean(sim))
    save_es_output("testout8_6.csv", es_absolute, es_diff_from_mean)
    return es_absolute, es_diff_from_mean


def main():
    # 8.1 = VaR from Normal Distribution
    test_81_var_normal()
    # 8.2 = VaR from T Distribution
    test_82_var_t()
    # 8.3 = VaR from Simulation
    test_83_var_simulation()
    # 8.4 = ES from Normal Distribution
    test_84_es_normal()
    # 8.5 = ES from T Distribution
    es85_abs, es85_diff = test_85_es_t()
    # 8.6 = ES from Simulation (compare to 8.5)
    es86_abs, es86_diff = test_86_es_simulation()

    print(f"8.5 ES Absolute      : {es85_abs:.10f}")
    print(f"8.6 ES Absolute      : {es86_abs:.10f}")
    print(f"Absolute difference  : {abs(es86_abs - es85_abs):.10f}")
    print(f"8.5 ES Diff from Mean: {es85_diff:.10f}")
    print(f"8.6 ES Diff from Mean: {es86_diff:.10f}")
    print(f"Diff-from-mean delta : {abs(es86_diff - es85_diff):.10f}")


if __name__ == "__main__":
    main()
