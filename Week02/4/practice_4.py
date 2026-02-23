import numpy as np
import pandas as pd


def chol_psd(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    # Cholesky-like factorization for PSD matrices (allows zero pivots)
    a = (a + a.T) / 2.0
    n = a.shape[0]
    l = np.zeros_like(a, dtype=float)

    for j in range(n):
        s = a[j, j] - np.dot(l[j, :j], l[j, :j])
        if s < 0.0 and s > -tol:
            s = 0.0
        if s < -tol:
            raise ValueError("Matrix is not PSD within tolerance.")

        l[j, j] = np.sqrt(s)
        if l[j, j] == 0.0:
            l[j + 1 :, j] = 0.0
            continue

        for i in range(j + 1, n):
            s = a[i, j] - np.dot(l[i, :j], l[j, :j])
            l[i, j] = s / l[j, j]

    return l


def main():
    cin = pd.read_csv("testout_3.1.csv").to_numpy(dtype=float)
    cout = chol_psd(cin)
    pd.DataFrame(cout).to_csv("testout_4.1.csv", index=False)


if __name__ == "__main__":
    main()
