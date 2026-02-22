## 1) Global Conventions (avoid silent point-loss)

- `alpha = 0.05` means **left-tail 5%**.
    
- Report VaR/ES as **positive loss** numbers (so bigger = worse).
    
- For matrices: shape usually `(n_assets, n_assets)`.
    
- For simulations: shape usually `(n_sim, n_assets)`.
    
- Always sanity check:
    
    - returns mean around 0 after de-meaning
        
    - covariance matrix symmetric
        
    - eigenvalues for PSD check
        

---

## 2) Decision Tree: which workflow is this prompt?

Use this like a checklist:

### A — Univariate stats + model choice

If prompt says: “mean/variance/skew/kurtosis”, “Normal vs t”, “AICc”, “fit distribution”  
➡️ **Workflow A**

### B — Univariate VaR/ES

If prompt says: “VaR/ES at 5%”, “parametric vs empirical”, “compare Normal vs t”  
➡️ **Workflow B**

### C — EW covariance / correlation (RiskMetrics style)

If prompt says: “EW covariance”, “lambda”, “recent weighting”, “RiskMetrics”  
➡️ **Workflow C**

### D — Missing data cov + PSD/PD + Higham + PCA variance table

If prompt says: “pairwise covariance”, “missing values”, “not PSD”, “Higham”, “PCA explained variance”  
➡️ **Workflow D**

### E — Copula simulation + portfolio $VaR/$ES

If prompt says: “fit t marginals”, “Gaussian copula”, “Spearman”, “simulate joint returns”, “compute portfolio VaR/ES”  
➡️ **Workflow E**

### F — Cholesky vs PCA simulation comparison

If prompt says: “simulate via Cholesky and PCA”, “compare speed/error”, “Frobenius norm”  
➡️ **Workflow F**

---

## 3) Input/Output Glue Snippets (stop wasting time)

### 3.1 Load a _returns vector_ from CSV (1 column or detect numeric)
```python
def load_returns_vector(csv_path):  
    df = pd.read_csv(csv_path)  
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()  
    if len(num_cols) == 0:  
        raise ValueError("No numeric column found.")  
    x = df[num_cols[0]].dropna().to_numpy()  
    return x
```
### 3.2 Load a _returns matrix_ (multi-asset, allow NaNs)
```python
def load_returns_matrix(csv_path):  
    df = pd.read_csv(csv_path)  
    num = df.select_dtypes(include=[np.number])  
    X = num.to_numpy()  # keep NaNs  
    cols = num.columns.tolist()  
    return X, cols
```
### 3.3 Load _prices_ with Date column (sorted)
```python
def load_prices(csv_path, date_col="Date"):  
    df = pd.read_csv(csv_path)  
    if date_col in df.columns:  
        df[date_col] = pd.to_datetime(df[date_col])  
        df = df.sort_values(date_col)  
    return df
```
### 3.4 Demean each column (required in many prompts)
```python
def demean_matrix(X):  
    mu = np.nanmean(X, axis=0)  
    return X - mu
```
### 3.5 PSD test (and symmetrize first)
```python
def symm(A):  
    return 0.5 * (A + A.T)  
  
def min_eig(A):  
    vals = np.linalg.eigvalsh(symm(A))  
    return float(vals.min()), vals
```
### 3.6 Empirical VaR/ES from a PnL sample (PnL: + = gain, - = loss)
```python
def var_es_from_pnl(pnl, alpha=0.05):  
    pnl = np.asarray(pnl)  
    q = np.quantile(pnl, alpha)     # left tail quantile (likely negative)  
    var = -q                        # report as positive loss  
    es = -pnl[pnl <= q].mean()      # mean of worst alpha tail, positive loss  
    return var, es
```
---

## 4) WORKFLOWS (the exam pipelines)

Each workflow below has:

- **When to use**
    
- **Steps**
    
- **Copy/paste skeleton**
    
- **What to report**
    
- **Wording template** (easy points)
    

---

# Workflow A — Univariate moments + Normal vs t (AICc)

### When to use

- “Compute mean/variance/skew/kurtosis”
    
- “Fit Normal and t, choose best with AICc”
    

### Steps

1. load returns vector `x`
    
2. moments: mean/var/skew/excess kurt
    
3. fit Normal vs t by AICc
    
4. report params + decision
    

### Skeleton
```python
alpha = 0.05  
x = load_returns_vector("YOURFILE.csv")  
  
m, v, s, ek = chapter1.first4_moments(x)  
fit = chapter1.fit_normal_vs_t_aicc(x)  
  
print("Moments:")  
print({"mean": m, "var": v, "skew": s, "excess_kurt": ek})  
print("Fit summary:", fit)  
print("Best model:", fit["best_model"])
```

### Report (numbers)

- mean, variance, skew, excess kurtosis
    
- Normal: mu, sigma, AICc
    
- t: nu, mu, sigma, AICc
    
- best_model
    

### Wording template (paste + tweak)

- “Skewness measures asymmetry; here skew is {s:.3f}, so the distribution is {roughly symmetric / skewed}.”
    
- “Excess kurtosis measures tail heaviness relative to Normal; {ek:.3f} suggests {heavier / lighter} tails.”
    
- “I fit Normal and Student-t by MLE and compare using AICc (small-sample corrected). The lower AICc indicates better out-of-sample fit; here {best_model} is preferred.”
    

---

# Workflow B — Univariate VaR/ES (Normal vs t vs empirical)

### When to use

- “Compute 5% VaR/ES”
    
- “Compare VaR/ES under Normal and t”
    
- “Explain why ES differs”
    

### Steps (parametric)

1. fit Normal/t
    
2. compute VaR/ES in return units
    
3. explain: ES is tail-mean beyond VaR
    

### Skeleton (robust: use your week05 ES functions)
```python
alpha = 0.05  
x = load_returns_vector("YOURFILE.csv")  
  
fit = chapter1.fit_normal_vs_t_aicc(x)  
  
# empirical:  
emp_var = -np.quantile(x, alpha)  
emp_es  = -x[x <= np.quantile(x, alpha)].mean()  
  
print("Empirical VaR/ES:", emp_var, emp_es)  
  
# parametric Normal (if you want):  
muN, sigN = fit["mu_norm"], fit["sigma_norm"]  
z = stats.norm.ppf(alpha)  
varN = -(muN + sigN * z)  
esN  = -muN + sigN * stats.norm.pdf(z) / alpha  
print("Normal VaR/ES:", varN, esN)  
  
# parametric t (use your chapter5 if available):  
nu, muT, sigT = fit["nu_t"], fit["mu_t"], fit["sigma_t"]  
varT, esT = chapter5.calculate_es_t(nu, muT, sigT, alpha=alpha)  
print("t VaR/ES:", varT, esT)
```
### Report

- VaR(5%), ES(5%) for empirical + (Normal, t if requested)
    
- Clear sign convention: “positive loss”
    

### Wording template

- “VaR at 5% is the loss threshold such that only 5% of outcomes are worse.”
    
- “ES at 5% is the **average loss** in that worst 5%, so it is more sensitive to tail thickness than VaR.”
    
- “If the data have heavy tails (excess kurtosis > 0), the t model typically increases ES more than VaR because it puts more probability mass into extreme losses.”
    

---

# Workflow C — EW covariance/correlation (RiskMetrics)

### When to use

- “EW covariance with lambda …”
    
- “EW correlation”
    
- “combine EW corr + EW var into cov”
    

### Steps

1. load returns matrix `X` (demean if asked)
    
2. compute EW covariance with lambda
    
3. optionally convert to corr or combine corr + variances
    

### Skeleton
```python
X, cols = load_returns_matrix("YOURFILE.csv")  
X = demean_matrix(X)  # if prompt says demean  
  
lam = 0.97  
ew_cov = chapter3.ew_covariance(X, lam=lam)  
  
print("EW covariance:\n", ew_cov)
```
### Wording template

- “Exponentially weighted estimates emphasize recent observations, capturing time-varying volatility/correlation (RiskMetrics idea).”
    
- “Lambda controls decay: higher lambda gives slower decay (longer memory), lower lambda reacts faster to new information.”
    

---

# Workflow D — Missing data covariance → PSD test → Higham fix → PCA variance table

### When to use

- “pairwise covariance” (skip_miss=False)
    
- “not PSD / negative eigenvalues”
    
- “fix with Higham”
    
- “PCA explained variance table”
    

### Steps

1. load returns matrix with NaNs
    
2. compute pairwise covariance (`skip_miss=False`)
    
3. check min eigenvalue
    
4. Higham nearest PSD
    
5. PCA cumulative explained variance
    

### Skeleton
```python
X, cols = load_returns_matrix("YOURFILE.csv")  
X = demean_matrix(X)  # often required  
  
cov_pair = chapter3.missing_cov(X, skip_miss=False)  
cov_pair = symm(cov_pair)  
  
m0, _ = min_eig(cov_pair)  
print("Min eigen (pairwise cov):", m0)  
  
cov_fix = chapter3.higham_nearest_psd(cov_pair)  
m1, _ = min_eig(cov_fix)  
print("Min eigen (Higham fixed):", m1)  
  
cum = chapter3.pca_cumulative_variance(cov_fix)  
  
pca_table = pd.DataFrame({  
    "k": np.arange(1, len(cum)+1),  
    "cum_explained": cum  
})  
print(pca_table.head(10))
```
### Report

- min eigenvalue before and after fix
    
- statement: “pairwise cov may be non-PSD”
    
- PCA cumulative explained variance (first 10, or until 75/90/95%)
    

### Wording template

- “Pairwise covariance uses different sample sets for different pairs, which can break the global consistency conditions required for a covariance matrix, so negative eigenvalues can appear.”
    
- “Higham’s algorithm projects onto the PSD cone (and constraints like symmetry/unit diagonal as needed) to find the nearest valid covariance/correlation matrix.”
    
- “PCA eigenvalues represent variance explained by orthogonal factors; cumulative explained variance indicates effective dimension.”
    

---

# Workflow E — Copula simulation → per-asset and total portfolio $VaR/$ES

### When to use

- “fit t marginals”
    
- “Gaussian copula”
    
- “Spearman dependence”
    
- “simulate joint returns”
    
- “compute VaR/ES for each asset and portfolio”
    

### Steps

1. prices → returns
    
2. demean returns if asked
    
3. simulate with copula
    
4. convert simulated returns into simulated PnL ($) using holdings
    
5. compute VaR/ES from PnL samples
    

### Skeleton
```python
alpha = 0.05  
prices = load_prices("PRICES.csv", date_col="Date")  
  
rets = chapter4.return_calculate(prices, date_column="Date", method="discrete")  
R = rets.select_dtypes(include=[np.number]).to_numpy()  
R = demean_matrix(R)  # if prompt says demean  
  
simR, cop_corr, marg = chapter5.simulate_copula_mixed(  
    R,  
    dist_types=["T"] * R.shape[1],   # or ["Normal","T","T"] if specified  
    n_sim=10000,  
    method="spearman",  
    seed=42  
)  
  
# holdings glue  
shares = np.array([100, 200, 150])  # <-- replace with prompt  
cur_px = prices.select_dtypes(include=[np.number]).iloc[-1].to_numpy()  
# PnL per asset: (sim return) * (shares*price)  
asset_pnl = simR * (shares * cur_px)  
port_pnl = asset_pnl.sum(axis=1)  
  
# VaR/ES  
port_var, port_es = var_es_from_pnl(port_pnl, alpha=alpha)  
  
print("Portfolio VaR/ES:", port_var, port_es)  
  
# Per-asset VaR/ES if asked  
for j in range(asset_pnl.shape[1]):  
    vj, ej = var_es_from_pnl(asset_pnl[:, j], alpha=alpha)  
    print(f"Asset {j} VaR/ES:", vj, ej)
```
### Report

- copula correlation matrix used (and method: Spearman/Pearson)
    
- VaR/ES per asset and portfolio (in $)
    
- mention: marginals fitted (t vs Normal)
    

### Wording template

- “I model each asset’s marginal distribution (often heavy-tailed via t) and model dependence separately via a Gaussian copula.”
    
- “Spearman correlation is rank-based and more robust to heavy tails/outliers than Pearson when fitting dependence.”
    
- “After simulating joint returns, I reprice holdings to generate a PnL distribution, then compute VaR/ES empirically from that simulated PnL.”
    

---

# Workflow F — Cholesky vs PCA simulation comparison

### When to use

- “compare simulation methods”
    
- “speed vs accuracy”
    
- “Frobenius norm error”
    

### Skeleton
```python
cov = ...  # from missing_cov or ew_cov  
out = chapter3.benchmark_cholesky_vs_pca(cov, n_simulations=10000, pct_exp=0.75, seed=42)  
  
print("Times:", out["time_cholesky"], out["time_pca"])  
print("Frob errors:", out["frob_error_cholesky"], out["frob_error_pca"])
```
### Wording template

- “Cholesky preserves the full covariance structure (higher fidelity) but can be slower in high dimensions.”
    
- “PCA uses a low-rank approximation capturing a target explained variance (faster) at the cost of covariance approximation error.”