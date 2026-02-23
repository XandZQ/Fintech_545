import numpy as np, pandas as pd
from qrm_lib import chapter1, chapter2, chapter3, chapter4, chapter5
from qrm_lib import exam_utils as eu

# --- (a) Load prices → returns ---
prices_df = eu.load_prices("midterm/problem4.csv", date_col="Date")
R, asset_names, last_prices = eu.prices_to_returns(prices_df, method="discrete", date_col="Date")
print(f"Returns shape: {R.shape}")
print(f"Asset names: {asset_names}")
print(f"Last prices: {last_prices}")

# --- (b) Fit marginals for each asset ---
print("\n=== Marginal Fits ===")
for j, name in enumerate(asset_names):
    r = R[:, j]
    res = eu.pipeline_moments_and_fit(r)
    print(f"\n{name}:")
    eu.print_moments(res)


# --- (c) Copula → Portfolio VaR/ES (single alpha) ---
alpha = 0.05       # ← 看题目
holdings = np.array([100, 200, 150, 100, 50])  # ← 看题目!!! 这个一定要从题目读
dist_types = ["t", "t", "t", "t", "t"]         # ← 看题目，可能 mix Normal/t

res_e = eu.pipeline_copula_portfolio(
    returns=R,
    holdings=holdings,
    current_prices=last_prices,
    dist_types=dist_types,
    alpha=alpha,
    n_sim=10000,
    method="spearman",
    seed=42,
    do_demean=True,
)
eu.print_portfolio_risk(res_e)

# --- (d) Multi-alpha version ---
alphas = [0.05, 0.01]  # ← 看题目
res_e2 = eu.pipeline_copula_multi_alpha(
    returns=R,
    holdings=holdings,
    current_prices=last_prices,
    alphas=alphas,
    dist_types=dist_types,
    n_sim=10000,
    method="spearman",
    seed=42,
    do_demean=True,
)
eu.print_multi_alpha_risk(res_e2)