import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import gammaln
from pathlib import Path


# Data containers 
@dataclass
class TParams:
    nu: float

@dataclass
class ErrorModel:
    mu: float
    sigma: float
    rho: TParams  # contains nu

@dataclass
class FitResult:
    errorModel: ErrorModel
    beta: np.ndarray  # regression coefficients (including intercept); empty for pure dist fits



# Logpdf for generalized Student-t: t(nu, (mu, sigma))

def logpdf_general_t(x, mu, sigma, nu):
    z = (x - mu) / sigma
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - np.log(sigma)
        - 0.5 * (np.log(np.pi) + np.log(nu))
        - ((nu + 1.0) / 2.0) * np.log1p((z * z) / nu)
    )



# 7.1 Fit Normal (MLE)

def fit_normal(x: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float).ravel()
    mu = float(np.mean(x))
    sigma = float(np.sqrt(np.mean((x - mu) ** 2)))  # MLE
    em = ErrorModel(mu=mu, sigma=sigma, rho=TParams(nu=np.nan))
    return FitResult(errorModel=em, beta=np.array([]))



# 7.2 Fit generalized Student-t via MLE over (mu, sigma, nu)
def fit_general_t(x: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float).ravel()
    mu0 = float(np.mean(x))
    sigma0 = float(np.sqrt(np.mean((x - mu0) ** 2)))
    nu0 = 10.0

    def nll(theta):
        mu = theta[0]
        sigma = np.exp(theta[1])
        nu = 2.0 + np.exp(theta[2])
        return float(-np.sum(logpdf_general_t(x, mu, sigma, nu)))

    theta0 = np.array([mu0, np.log(sigma0 + 1e-12), np.log(nu0 - 2.0)], dtype=float)
    res = minimize(nll, theta0, method="BFGS")

    mu_hat = float(res.x[0])
    sigma_hat = float(np.exp(res.x[1]))
    nu_hat = float(2.0 + np.exp(res.x[2]))

    em = ErrorModel(mu=mu_hat, sigma=sigma_hat, rho=TParams(nu=nu_hat))
    return FitResult(errorModel=em, beta=np.array([]))



# 7.3 Fit t regression:
def fit_regression_t(y: np.ndarray, X: np.ndarray) -> FitResult:
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = y.shape[0]
    if X.shape[0] != n:
        raise ValueError("X and y must have same number of rows")

    # Add intercept
    Xtilde = np.column_stack([np.ones(n), X])
    p = Xtilde.shape[1]  # 1 + num_features (should be 4 for x1,x2,x3)

    # OLS start
    beta0, *_ = np.linalg.lstsq(Xtilde, y, rcond=None)
    resid0 = y - Xtilde @ beta0
    sigma0 = float(np.sqrt(np.mean(resid0 ** 2)))
    nu0 = 10.0

    # theta = [beta (p), log_sigma, log(nu-2)]
    def nll(theta):
        beta = theta[:p]
        sigma = np.exp(theta[p])
        nu = 2.0 + np.exp(theta[p + 1])
        r = y - Xtilde @ beta
        # errors have mu=0
        return float(-np.sum(logpdf_general_t(r, 0.0, sigma, nu)))

    theta0 = np.concatenate([beta0, [np.log(sigma0 + 1e-12), np.log(nu0 - 2.0)]])
    res = minimize(nll, theta0, method="BFGS")

    beta_hat = res.x[:p].astype(float)
    sigma_hat = float(np.exp(res.x[p]))
    nu_hat = float(2.0 + np.exp(res.x[p + 1]))

    em = ErrorModel(mu=0.0, sigma=sigma_hat, rho=TParams(nu=nu_hat))
    return FitResult(errorModel=em, beta=beta_hat)



def main():
    # Path to Week01 directory 
    work_dir = Path(__file__).resolve().parent

    # Path to input data 
    root = work_dir.parent
    data_dir = root / "testfiles" / "data"

    # 7.1 Normal 
    cin = pd.read_csv(data_dir / "test7_1.csv")
    fd = fit_normal(cin.iloc[:, 0].to_numpy())
    pd.DataFrame(
        {
            "mu": [fd.errorModel.mu],
            "sigma": [fd.errorModel.sigma],
        }
    ).to_csv(work_dir / "testout7_1.csv", index=False)

    # 7.2 Student-t
    cin = pd.read_csv(data_dir / "test7_2.csv")
    fd = fit_general_t(cin.iloc[:, 0].to_numpy())
    pd.DataFrame(
        {
            "mu": [fd.errorModel.mu],
            "sigma": [fd.errorModel.sigma],
            "nu": [fd.errorModel.rho.nu],
        }
    ).to_csv(work_dir / "testout7_2.csv", index=False)

    # 7.3 t Regression 
    cin = pd.read_csv(data_dir / "test7_3.csv")
    y = cin["y"].to_numpy()
    X = cin.drop(columns=["y"]).to_numpy()

    fd = fit_regression_t(y, X)

    pd.DataFrame(
        {
            "mu": [fd.errorModel.mu],
            "sigma": [fd.errorModel.sigma],
            "nu": [fd.errorModel.rho.nu],
            "Alpha": [fd.beta[0]],
            "B1": [fd.beta[1]],
            "B2": [fd.beta[2]],
            "B3": [fd.beta[3]],
        }
    ).to_csv(work_dir / "testout7_3.csv", index=False)

    print("Outputs written to Week01/")


if __name__ == "__main__":
    main()
