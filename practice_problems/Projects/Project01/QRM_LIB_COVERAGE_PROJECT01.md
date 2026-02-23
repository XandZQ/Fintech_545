# Project01 Coverage vs `qrm_lib`

This file maps `project1_code.ipynb` answer code to the prep library.

Source notebook: `practice_problems/Projects/Project01/project1_code.ipynb`

## Problem 1
- Notebook cells:
  - `Cell 5`: mean/variance/skew/kurtosis
  - `Cell 8`: AICc + Normal fit + t fit
- `qrm_lib` mapping:
  - `qrm_lib/chapter1.py::first4_moments`
  - `qrm_lib/chapter1.py::calculate_aicc`
  - `qrm_lib/chapter1.py::fit_normal_vs_t_aicc`
- Coverage: `Covered`
- Notes:
  - Notebook uses `np.var` (biased by default). Library uses unbiased variance in `first4_moments`.

## Problem 2
- Notebook cells:
  - `Cell 12`: pairwise covariance
  - `Cell 14`: PSD eigenvalue check
  - `Cell 16`: Higham nearest PSD
  - `Cell 17`: Rebonato-Jackel near PSD
  - `Cell 19`: overlap-only covariance
- `qrm_lib` mapping:
  - `qrm_lib/chapter3.py::higham_nearest_psd`
  - `qrm_lib/chapter3.py::near_psd`
  - `qrm_lib/chapter3.py::missing_cov` (`skip_miss=True` for overlap-only, `False` for pairwise)
- Coverage: `Covered`

## Problem 3
- Notebook cells:
  - `Cell 22`: estimate mean/cov from data
  - `Cell 24`: Cholesky conditional simulation for `X2 | X1=0.6`
- `qrm_lib` mapping:
  - `qrm_lib/chapter3.py::conditional_bivariate_stats` (closed-form conditional mean/var)
  - `qrm_lib/chapter3.py::simulate_conditional_bivariate` (Cholesky simulation proof)
  - `qrm_lib/chapter3.py::simulate_normal_cholesky` (general MVN simulation utility)
- Coverage: `Covered`

## Problem 4
- Notebook cells:
  - `Cell 28`: MA(1..3) model diagnostics (ACF/PACF on residuals)
  - `Cell 30`: AR(1..3) model diagnostics (ACF/PACF on residuals)
  - `Cell 34`: AR(1..10) AICc selection
- `qrm_lib` mapping:
  - `qrm_lib/chapter2.py::fit_ma_model`
  - `qrm_lib/chapter2.py::fit_ar_model`
  - `qrm_lib/chapter2.py::model_result_aicc`
  - `qrm_lib/chapter2.py::scan_ar_orders_aicc`
  - `qrm_lib/chapter2.py::select_best_ar_order_aicc`
  - `qrm_lib/chapter2.py::simulate_ma_process`, `simulate_ma_orders` (added for MA(q) simulation)
  - `qrm_lib/chapter2.py::simulate_ar_process`, `simulate_ar_orders` (added for AR(p) simulation)
  - `qrm_lib/chapter2.py::plot_acf_pacf_grid` (added for ACF/PACF batch plotting)
- Coverage: `Covered`
- Notes:
  - Notebook focuses on fitted-model residual diagnostics; prompt also asks simulated processes.
  - New simulation helpers close that prompt gap.

## Problem 5
- Notebook cells:
  - `Cell 38`: exponential weights + EW covariance
  - `Cell 39-41`: PCA cumulative variance vs lambda
- `qrm_lib` mapping:
  - `qrm_lib/chapter3.py::exponential_weights`
  - `qrm_lib/chapter3.py::ew_covariance`
  - `qrm_lib/chapter3.py::pca_cumulative_variance`
- Coverage: `Covered`

## Problem 6
- Notebook cells:
  - `Cell 45`: Cholesky simulation
  - `Cell 47`: PCA simulation (75% explained variance)
  - `Cell 49, 51, 53`: Frobenius comparison, explained variance curves, timing
- `qrm_lib` mapping:
  - `qrm_lib/chapter3.py::simulate_normal_cholesky`
  - `qrm_lib/chapter3.py::simulate_pca` (`pct_exp=0.75`)
  - `qrm_lib/chapter3.py::benchmark_cholesky_vs_pca`
  - `qrm_lib/chapter3.py::pca_cumulative_variance`
- Coverage: `Covered`

## Final Summary
- Coverage against Project01 notebook code: `Fully covered`.
- New additions made to complete coverage:
  - `qrm_lib/chapter2.py::simulate_ar_process`
  - `qrm_lib/chapter2.py::simulate_ma_process`
  - `qrm_lib/chapter2.py::simulate_ar_orders`
  - `qrm_lib/chapter2.py::simulate_ma_orders`
  - `qrm_lib/chapter2.py::plot_acf_pacf_grid`
