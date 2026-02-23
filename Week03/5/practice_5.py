import numpy as np
import pandas as pd


def make_symmetric(matrix):
    return (matrix + matrix.T) / 2.0


def nearest_psd_by_clipping(matrix):
    """
    near_psd:
        Symmetrize
        Set negative eigenvalues to zero
        Rescale to keep original diagonal
    """
    matrix = make_symmetric(matrix)

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < 0.0] = 0.0
    fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T

    old_diag = np.diag(matrix)
    new_diag = np.diag(fixed)
    scale = np.ones_like(old_diag)

    for i in range(len(scale)):
        if new_diag[i] > 0.0:
            scale[i] = np.sqrt(old_diag[i] / new_diag[i])

    d = np.diag(scale)
    fixed = d @ fixed @ d
    return make_symmetric(fixed)


def project_to_psd(matrix):
    matrix = make_symmetric(matrix)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < 0.0] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def nearest_psd_higham(matrix, max_iter=100, tol=1e-8):
    """
    Higham nearest PSD using alternating projections.
    Keeps the original diagonal.
    """
    matrix = make_symmetric(matrix)
    target_diag = np.diag(matrix).copy()

    y = matrix.copy()
    delta_s = np.zeros_like(matrix)

    for _ in range(max_iter):
        r = y - delta_s
        x = project_to_psd(r)
        delta_s = x - r

        y = x.copy()
        np.fill_diagonal(y, target_diag)
        y = make_symmetric(y)

        rel_change = np.linalg.norm(y - x, ord="fro") / max(1.0, np.linalg.norm(y, ord="fro"))
        if rel_change < tol:
            break

    return make_symmetric(project_to_psd(y))


def psd_square_root(matrix, tol=1e-10):
    """
    Build a square-root matrix L so that L @ L.T is approximately matrix.
    Works for PSD matrices.
    """
    matrix = make_symmetric(matrix)
    eigvals, eigvecs = np.linalg.eigh(matrix)

    if np.min(eigvals) < -tol:
        raise ValueError("Matrix is not PSD.")

    eigvals[eigvals < 0.0] = 0.0
    sqrt_diag = np.diag(np.sqrt(eigvals))
    return eigvecs @ sqrt_diag


def simulate_normal_with_cov(cov_matrix, n_samples):
    l = psd_square_root(cov_matrix)
    z = np.random.normal(size=(n_samples, cov_matrix.shape[0]))
    return z @ l.T


def simulate_pca(cov_matrix, n_samples, pct_explained=0.99):
    """
    PCA simulation:
    keep the top components that explain >= pct_explained variance.
    """
    cov_matrix = make_symmetric(cov_matrix)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals[eigvals < 0.0] = 0.0

    total_var = np.sum(eigvals)
    if total_var <= 0.0:
        return np.zeros((n_samples, cov_matrix.shape[0]))

    explained = np.cumsum(eigvals) / total_var
    k = np.searchsorted(explained, pct_explained) + 1

    loadings = eigvecs[:, :k] @ np.diag(np.sqrt(eigvals[:k]))
    z = np.random.normal(size=(n_samples, k))
    return z @ loadings.T


def save_covariance_from_sim(sim_data, columns, out_file):
    cov = np.cov(sim_data, rowvar=False)
    pd.DataFrame(cov, columns=columns).to_csv(out_file, index=False)


def main():
    np.random.seed(4)

    n_samples = 100000
    columns = [f"x{i}" for i in range(1, 6)]

    # 5.1 PD input -> normal simulation
    cov_51 = pd.read_csv("test5_1.csv").to_numpy(dtype=float)
    sim_51 = simulate_normal_with_cov(cov_51, n_samples)
    save_covariance_from_sim(sim_51, columns, "testout_5.1.csv")

    # 5.2 PSD input -> normal simulation
    cov_52 = pd.read_csv("test5_2.csv").to_numpy(dtype=float)
    sim_52 = simulate_normal_with_cov(cov_52, n_samples)
    save_covariance_from_sim(sim_52, columns, "testout_5.2.csv")

    # 5.3 non-PSD input -> near_psd fix -> normal simulation
    cov_53 = pd.read_csv("test5_3.csv").to_numpy(dtype=float)
    fixed_53 = nearest_psd_by_clipping(cov_53)
    sim_53 = simulate_normal_with_cov(fixed_53, n_samples)
    save_covariance_from_sim(sim_53, columns, "testout_5.3.csv")

    # 5.4 non-PSD input -> Higham fix -> normal simulation
    cov_54 = pd.read_csv("test5_3.csv").to_numpy(dtype=float)
    fixed_54 = nearest_psd_higham(cov_54)
    sim_54 = simulate_normal_with_cov(fixed_54, n_samples)
    save_covariance_from_sim(sim_54, columns, "testout_5.4.csv")

    # 5.5 PSD input -> PCA simulation (99% explained)
    cov_55 = pd.read_csv("test5_2.csv").to_numpy(dtype=float)
    sim_55 = simulate_pca(cov_55, n_samples, pct_explained=0.99)
    save_covariance_from_sim(sim_55, columns, "testout_5.5.csv")


if __name__ == "__main__":
    main()
