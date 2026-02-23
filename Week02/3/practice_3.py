import numpy as np
import pandas as pd


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2.0


def _proj_sdp(a: np.ndarray) -> np.ndarray:
    # Projection onto PSD cone via eigenvalue clipping
    vals, vecs = np.linalg.eigh(_symmetrize(a))
    vals[vals < 0.0] = 0.0
    return vecs @ np.diag(vals) @ vecs.T


def _proj_diag(a: np.ndarray, target_diag: np.ndarray) -> np.ndarray:
    b = a.copy()
    np.fill_diagonal(b, target_diag)
    return b


def near_psd(a: np.ndarray) -> np.ndarray:
    # PSD by eigenvalue clipping + rescale to match original diagonal
    a = _symmetrize(a)
    vals, vecs = np.linalg.eigh(a)
    vals[vals < 0.0] = 0.0
    b = vecs @ np.diag(vals) @ vecs.T

    orig_diag = np.diag(a)
    new_diag = np.diag(b)
    scale = np.ones_like(orig_diag)
    mask = new_diag > 0.0
    scale[mask] = np.sqrt(orig_diag[mask] / new_diag[mask])
    d = np.diag(scale)
    return d @ b @ d


def higham_nearest_psd(
    a: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> np.ndarray:
    # Higham alternating projections onto PSD + fixed diagonal
    a = _symmetrize(a)
    target_diag = np.diag(a).copy()
    y = a.copy()
    delta_s = np.zeros_like(a)

    for _ in range(max_iter):
        r = y - delta_s
        x = _proj_sdp(r)
        delta_s = x - r
        y = _proj_diag(x, target_diag)
        y = _symmetrize(y)

        diff = np.linalg.norm(y - x, ord="fro") / max(1.0, np.linalg.norm(y, ord="fro"))
        if diff < tol:
            break

    return _proj_sdp(y)


def main():
    # 3.1 near_psd covariance
    cin = pd.read_csv("testout_1.3.csv").to_numpy(dtype=float)
    cout = near_psd(cin)
    pd.DataFrame(cout).to_csv("testout_3.1.csv", index=False)

    # 3.2 near_psd correlation
    cin = pd.read_csv("testout_1.4.csv").to_numpy(dtype=float)
    cout = near_psd(cin)
    pd.DataFrame(cout).to_csv("testout_3.2.csv", index=False)

    # 3.3 Higham covariance
    cin = pd.read_csv("testout_1.3.csv").to_numpy(dtype=float)
    cout = higham_nearest_psd(cin)
    pd.DataFrame(cout).to_csv("testout_3.3.csv", index=False)

    # 3.4 Higham correlation
    cin = pd.read_csv("testout_1.4.csv").to_numpy(dtype=float)
    cout = higham_nearest_psd(cin)
    pd.DataFrame(cout).to_csv("testout_3.4.csv", index=False)


if __name__ == "__main__":
    main()
