import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t


N_SIM = 100000
ALPHA = 0.05


def fit_normal(x):
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    return {"dist": "Normal", "mean": mean, "std": std}


def fit_t(x):
    df, loc, scale = t.fit(x)
    return {"dist": "T", "df": df, "loc": loc, "scale": scale}


def to_uniform(x, model):
    if model["dist"] == "Normal":
        return norm.cdf(x, loc=model["mean"], scale=model["std"])
    return t.cdf(x, df=model["df"], loc=model["loc"], scale=model["scale"])


def from_uniform(u, model):
    if model["dist"] == "Normal":
        return norm.ppf(u, loc=model["mean"], scale=model["std"])
    return t.ppf(u, df=model["df"], loc=model["loc"], scale=model["scale"])


def simulate_gaussian_copula_u(corr_matrix, n_sim):
    # Use eigen-decomposition to build correlated normal draws
    eigvals, eigvecs = np.linalg.eigh((corr_matrix + corr_matrix.T) / 2.0)
    eigvals[eigvals < 0.0] = 0.0
    loadings = eigvecs @ np.diag(np.sqrt(eigvals))

    z = np.random.normal(size=(n_sim, corr_matrix.shape[0]))
    correlated_z = z @ loadings.T
    return norm.cdf(correlated_z)


def var_95(pnl):
    return -np.quantile(pnl, ALPHA)


def es_95(pnl):
    q = np.quantile(pnl, ALPHA)
    return -np.mean(pnl[pnl <= q])


def main():
    returns_df = pd.read_csv("test9_1_returns.csv")
    portfolio_df = pd.read_csv("test9_1_portfolio.csv")

    # Fit marginal distributions
    model_a = fit_normal(returns_df["A"].to_numpy(dtype=float))
    model_b = fit_t(returns_df["B"].to_numpy(dtype=float))

    # Build Spearman correlation on transformed uniforms
    u_a = to_uniform(returns_df["A"].to_numpy(dtype=float), model_a)
    u_b = to_uniform(returns_df["B"].to_numpy(dtype=float), model_b)
    spearman_corr = spearmanr(u_a, u_b).correlation
    corr_matrix = np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])

    # Simulate copula uniforms and map back to each marginal
    u_sim = simulate_gaussian_copula_u(corr_matrix, N_SIM)
    sim_a = from_uniform(u_sim[:, 0], model_a)
    sim_b = from_uniform(u_sim[:, 1], model_b)
    sim_returns = {"A": sim_a, "B": sim_b}

    # Compute stock-level PnL paths and risk
    rows = []
    total_pnl = np.zeros(N_SIM)
    total_current_value = 0.0

    for _, row in portfolio_df.iterrows():
        stock = row["Stock"]
        holding = float(row["Holding"])
        start_price = float(row["Starting Price"])
        current_value = holding * start_price

        simulated_value = current_value * (1.0 + sim_returns[stock])
        pnl = simulated_value - current_value
        total_pnl += pnl
        total_current_value += current_value

        stock_var = var_95(pnl)
        stock_es = es_95(pnl)

        rows.append(
            {
                "Stock": stock,
                "VaR95": stock_var,
                "ES95": stock_es,
                "VaR95_Pct": stock_var / current_value,
                "ES95_Pct": stock_es / current_value,
            }
        )

    # Portfolio total row (risk of combined PnL, not sum of individual VaR/ES)
    total_var = var_95(total_pnl)
    total_es = es_95(total_pnl)
    rows.append(
        {
            "Stock": "Total",
            "VaR95": total_var,
            "ES95": total_es,
            "VaR95_Pct": total_var / total_current_value,
            "ES95_Pct": total_es / total_current_value,
        }
    )

    out = pd.DataFrame(rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    out.to_csv("testout9_1.csv", index=False)


if __name__ == "__main__":
    main()
