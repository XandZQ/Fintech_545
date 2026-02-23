import numpy as np, pandas as pd, scipy.stats as stats, sys
from qrm_lib import chapter1, chapter2, chapter3, chapter4, chapter5
from qrm_lib import exam_utils as eu

# ============ PROBLEM 3 — PIPELINE VERSION ============
prices_df = eu.load_prices("MidTerm/problem4.csv", date_col="Date")
print(f"Shape: {prices_df.shape}")
print(f"Date range: {prices_df['Date'].iloc[0]} to {prices_df['Date'].iloc[-1]}")
print(f"Last prices:")
for col in prices_df.columns[1:]:
    print(f"  {col}: ${prices_df[col].iloc[-1]:.2f}")

# Discrete returns: r_t = (P_t - P_{t-1}) / P_{t-1}
rets_df = chapter4.return_calculate(prices_df, date_column="Date", method="discrete")
R = rets_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
asset_names = rets_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Returns shape: {R.shape}")
print(f"Assets: {asset_names}")

print(f"{'Asset':<8} {'Mean':>10} {'Std':>10} {'Skew':>8} {'ExKurt':>8}")
print("-" * 50)
for j, name in enumerate(asset_names):
    m, v, s, ek = chapter1.first4_moments(R[:, j])
    print(f"{name:<8} {m:>10.6f} {np.sqrt(v):>10.6f} {s:>8.4f} {ek:>8.4f}")

print("\n=== Distribution Fitting ===")
marginal_params = {}
for j, name in enumerate(asset_names):
    fit = chapter1.fit_normal_vs_t_aicc(R[:, j])
    best = fit["best_model"]
    print(f"{name}: best={best}, AICc_N={fit['aicc_norm']:.2f}, AICc_t={fit['aicc_t']:.2f}", end="")
    if best == "t":
        print(f", nu={fit['nu_t']:.2f}")
    else:
        print()
    marginal_params[name] = fit

R_dm = eu.demean(R)
dist_types = ["t", "t", "t", "t", "t"]  # ← 看题目
u_data, marginals = chapter5.fit_copula_marginals(R_dm, dist_types=dist_types)
print("Fitted marginals:")
for m in marginals:
    print(f"  {m}")

# Spearman (recommended)
cop_corr_sp = chapter5.fit_gaussian_copula_corr(u_data, method="spearman")
print("Spearman copula correlation:")
print(pd.DataFrame(cop_corr_sp, index=asset_names, columns=asset_names).round(4))

# Pearson (for comparison if asked)
cop_corr_pe = chapter5.fit_gaussian_copula_corr(u_data, method="pearson")
print("\nPearson copula correlation:")
print(pd.DataFrame(cop_corr_pe, index=asset_names, columns=asset_names).round(4))

n_sim = 10000  # ← 看题目
seed = 42      # ← 看题目

sim_R = chapter5.simulate_copula_from_fitted(
    marginals=marginals,
    corr=cop_corr_sp,
    n_sim=n_sim,
    seed=seed
)
print(f"Simulated returns shape: {sim_R.shape}")

holdings = np.array([100, 200, 150, 100, 50])  # ← 看题目!!!
last_prices = prices_df.select_dtypes(include=[np.number]).iloc[-1].to_numpy(dtype=float)

# Portfolio value
port_value = np.sum(holdings * last_prices)
print(f"Current portfolio value: ${port_value:,.2f}")

# Per-asset dollar positions
for j, name in enumerate(asset_names):
    pos = holdings[j] * last_prices[j]
    print(f"  {name}: {holdings[j]} shares × ${last_prices[j]:.2f} = ${pos:,.2f} ({pos/port_value*100:.1f}%)")

# PnL matrix
a_pnl = eu.asset_pnl_matrix(sim_R, holdings, last_prices)  # (n_sim, 5)
p_pnl = a_pnl.sum(axis=1)  # (n_sim,)
print(f"\nPnL statistics:")
print(f"  Mean: ${p_pnl.mean():,.2f}")
print(f"  Std:  ${p_pnl.std():,.2f}")
print(f"  Min:  ${p_pnl.min():,.2f}")
print(f"  Max:  ${p_pnl.max():,.2f}")

alpha = 0.05  # ← 看题目
port_var, port_es = eu.var_es_from_pnl(p_pnl, alpha=alpha)
print(f"Portfolio VaR({alpha}): ${port_var:,.2f}")
print(f"Portfolio ES({alpha}):  ${port_es:,.2f}")

print(f"\n{'Asset':<8} {'VaR':>12} {'ES':>12}")
print("-" * 34)
sum_var = 0
for j, name in enumerate(asset_names):
    vj, ej = eu.var_es_from_pnl(a_pnl[:, j], alpha=alpha)
    print(f"{name:<8} ${vj:>10,.2f} ${ej:>10,.2f}")
    sum_var += vj
print("-" * 34)
print(f"{'Sum':<8} ${sum_var:>10,.2f}")
print(f"{'Portfolio':<8} ${port_var:>10,.2f}")
print(f"{'Diversification benefit':<8}: ${sum_var - port_var:,.2f} ({(sum_var - port_var)/sum_var*100:.1f}%)")

alphas = [0.05, 0.01]  # ← 看题目
for a in alphas:
    v, e = eu.var_es_from_pnl(p_pnl, alpha=a)
    print(f"Alpha={a} ({(1-a)*100:.0f}%): VaR=${v:,.2f}, ES=${e:,.2f}")

# Spearman
sim_sp, _, _ = chapter5.simulate_copula_mixed(
    R_dm, dist_types=dist_types, n_sim=n_sim, method="spearman", seed=seed
)
pnl_sp = eu.portfolio_pnl(sim_sp, holdings, last_prices)
var_sp, es_sp = eu.var_es_from_pnl(pnl_sp, alpha=alpha)

# Pearson
sim_pe, _, _ = chapter5.simulate_copula_mixed(
    R_dm, dist_types=dist_types, n_sim=n_sim, method="pearson", seed=seed
)
pnl_pe = eu.portfolio_pnl(sim_pe, holdings, last_prices)
var_pe, es_pe = eu.var_es_from_pnl(pnl_pe, alpha=alpha)

print(f"Spearman: VaR=${var_sp:,.2f}, ES=${es_sp:,.2f}")
print(f"Pearson:  VaR=${var_pe:,.2f}, ES=${es_pe:,.2f}")