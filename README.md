# QRM Library

Python helpers for course 545 (Quantitative Risk Management), aligned to the chapter Julia notebooks.

## Quick Start
```python
from qrm_lib import chapter1, chapter2, chapter3, chapter4, chapter5
import numpy as np
import pandas as pd
```

## Fast Exam Lookup
- Open `qrm_lib/EXAM_QUICKREF.md` for:
  - what to import
  - which function to use by problem type
  - each function's input/output
  - copy-paste workflow templates

## Modules
- `qrm_lib/chapter1.py`: univariate stats, moments, normal PDF/CDF helpers
- `qrm_lib/chapter2.py`: correlation, MLE, regression, AR/MA simulation
- `qrm_lib/chapter3.py`: PSD fixes, missing-data covariance, PCA simulation
- `qrm_lib/chapter4.py`: return conversion, VaR methods, historical ES
- `qrm_lib/chapter5.py`: advanced ES, t fitting, Gaussian copula simulation
