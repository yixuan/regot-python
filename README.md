# RegOT-Python<img src="https://statr.me/images/sticker-regot.png" alt="regot" height="150px" align="right" />

**RegOT** is a collection of state-of-the-art solvers for
regularized optimal transport (OT) problems, implemented in
efficient C++ code. This repository is the Python interface
to **RegOT**.

## 📝 Formulation

**RegOT** mainly solves two types of regularized OT problems:
the entropic-regularized OT (EROT) and the quadratically regularized OT (QROT).

EROT, also known as the Sinkhorn-type OT, considers the following optimization problem:

```math
\begin{align*}
\min_{T\in\mathbb{R}^{n\times m}}\quad & \langle T,M\rangle-\eta h(T),\\
\text{subject to}\quad & T\mathbf{1}_{m}=a,T^{T}\mathbf{1}_{n}=b,T\ge0,
\end{align*}
```

where $a\in\mathbb{R}^n$ and $b\in\mathbb{R}^m$ are two given
probability vectors with $a_i>0$, $b_j>0$, $\sum_{i=1}^n a_i=\sum_{j=1}^m b_j=1$,
and $M\in\mathbb{R}^{n\times m}$ is a given cost matrix.
The function $h(T)=\sum_{i=1}^{n}\sum_{j=1}^{m}T_{ij}(1-\log T_{ij})$ is the entropy term,
and $\eta>0$ is a regularization parameter.

QROT, also known as the Euclidean-regularized OT, is concerned with the problem

```math
\begin{align*}
\min_{T\in\mathbb{R}^{n\times m}}\quad & \langle T,M\rangle+(\gamma/2) \Vert T \Vert_F^2,\\
\text{subject to}\quad & T\mathbf{1}_{m}=a,T^{T}\mathbf{1}_{n}=b,T\ge0.
\end{align*}
```

## 🔧 Solvers

Currently **RegOT** contains the following solvers for EROT (methods marked with 🌟 are developed by our group!):

- `sinkhorn_bcd`: the block coordinate descent (BCD) algorithm, equivalent to the well-known Sinkhorn algorithm.
- `sinkhorn_apdagd`: the adaptive primal-dual accelerate gradient descent (APDAGD) algorithm
([link to paper](https://arxiv.org/pdf/1802.04367)).
- `sinkhorn_lbfgs_dual`: the L-BFGS algorithm applied to the dual problem of EROT.
- `sinkhorn_newton`: Newton's method applied to the dual problem of EROT.
- 🌟`sinkhorn_sparse_newton`: Newton-type method using sparsified Hessian matrix, as described in [our SPLR paper](https://openreview.net/pdf?id=WCkMkMcqpb).
- 🌟`sinkhorn_ssns`: the safe and sparse Newton method for Sinkhorn-type OT (SSNS, [link to paper](https://openreview.net/pdf?id=Nmmiyjw7Xg)).
- 🌟`sinkhorn_splr`: the sparse-plus-low-rank quasi-Newton method for the dual problem of EROT (SPLR, [link to paper](https://openreview.net/pdf?id=WCkMkMcqpb)).

The following solvers are available for the QROT problem:

- `qrot_bcd`: the BCD algorithm.
- `qrot_gd`: the line search gradient descent algorithm applied to the dual problem of QROT.
- `qrot_apdagd`: the APDAGD algorithm ([link to paper](https://arxiv.org/pdf/1802.04367)).
- `qrot_pdaam`: the primal-dual accelerated alternating minimization (PDAAM) algorithm ([link to paper](https://arxiv.org/pdf/1906.03622)).
- `qrot_lbfgs_dual`: the L-BFGS algorithm applied to the dual problem of QROT.
- `qrot_lbfgs_semi_dual`: the L-BFGS algorithm applied to the semi-dual problem of QROT ([link to paper](https://arxiv.org/pdf/1710.06276)).
- `qrot_assn`: the adaptive semi-smooth Newton (ASSN) method applied to the dual problem of QROT ([link to paper](https://arxiv.org/pdf/1603.07870)).
- `qrot_grssn`: the globalized and regularized semi-smooth Newton (GRSSN) method applied to the dual problem of QROT ([link to paper](https://arxiv.org/pdf/1903.01112)).

All the solvers above return an object containing fields:

- `niter`: number of iterations used.
- `dual`: final dual variables.
- `plan`: computed transport plan.
- `obj_vals`: history of dual objective function values.
- `mar_errs`: history of marginal errors.
- `run_times`: cumulative runtimes of iterations in milliseconds.

A specialized primal-dual interior-point solver is also available for the QROT problem:

- `qrot_pdip`: the primal-dual interior-point method. Pass `inner_solver="cg"` for a CG-based inner solver, or `inner_solver="fp"` for the fixed-point method inner solver. Default is `inner_solver="cg"`.
- `pdip_cg` / `pdip_fp`: aliases for `qrot_pdip` with `inner_solver="cg"` and `"fp"`, respectively.
- For `cg`: default `tol=1e-8` for normalized primal/dual gaps and `mu`. When `cg_stop_gap_mu_only` is false (default), marginal-error stopping uses `cg_mar_tol` (default `1e-10`), independent of `tol`. Pass `cg_mar_tol` in kwargs to override. Set `cg_stop_gap_mu_only=True` to require only gaps + `mu` (like the FP solver).
- For `fp`: default `tol=1e-8`. By default, stopping uses only normalized primal/dual gaps and `mu` (`fp_stop_gap_mu_only=True`). Pass `fp_stop_gap_mu_only=False` if you also want to stop when marginal error `mar_err` falls below `tol`.

## 💽 Installation

### Using `pip`

You can simply install **RegOT** using the `pip` command:

```bash
pip install regot
```

### Building from source

A C++ compiler is needed to build **RegOT** from source. Enter the source directory and run

```bash
pip install . -r requirements.txt
```

### Developer / Profiling builds

An optional environment variable `REGOT_PDIP_DEV` can be set during installation (e.g., `REGOT_PDIP_DEV=1 pip install .`) to enable additional profiling functions. See `src/pdip_dev_flags.h` for details.

## 📗 Example

The code below shows minimal examples computing EROT and QROT transport plans given $a$, $b$, $M$, and $\eta$.

```py
import numpy as np
from scipy.stats import expon, norm
import regot
import matplotlib.pyplot as plt

# OT between two discretized distributions
# One is exponential, the other is mixture normal
def example(n=100, m=80):
    x1 = np.linspace(0.0, 5.0, num=n)
    x2 = np.linspace(0.0, 5.0, num=m)
    distr1 = expon(scale=1.0)
    distr2 = norm(loc=1.0, scale=0.2)
    distr3 = norm(loc=3.0, scale=0.5)
    a = distr1.pdf(x1)
    a = a / np.sum(a)
    b = 0.2 * distr2.pdf(x2) + 0.8 * distr3.pdf(x2)
    b = b / np.sum(b)
    M = np.square(x1.reshape(n, 1) - x2.reshape(1, m))
    return M, a, b

# Source and target distribution vectors `a` and `b`
# Cost matrix `M`
# Regularization parameter `reg`
np.random.seed(123)
M, a, b = example(n=100, m=80)
reg = 0.1

# EROT transport plans
# Algorithm: block coordinate descent (the Sinkhorn algorithm)
res1 = regot.sinkhorn_bcd(
    M, a, b, reg, tol=1e-6, max_iter=1000, verbose=1)

# Algorithm: SSNS
reg = 0.01
res2 = regot.sinkhorn_ssns(
    M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)

# QROT transport plans
res3 = regot.qrot_pdip(
    M, a, b, reg=0.1, tol=1e-8, max_iter=1000, inner_solver="cg")
res4 = regot.qrot_pdip(
    M, a, b, reg=0.01, tol=1e-8, max_iter=1000, inner_solver="fp")
```

We can retrieve the computed transport plans and visualize them using heatmaps:

```py
def vis_plan(T, title=""):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(T, interpolation="nearest")
    plt.title(title, fontsize=20)
    plt.show()

vis_plan(res1.plan, title="EROT (BCD), reg=0.1")
vis_plan(res2.plan, title="EROT (SSNS), reg=0.01")
vis_plan(res3.plan, title="QROT (PDIP-CG), reg=0.1")
vis_plan(res4.plan, title="QROT (PDIP-FP), reg=0.01")
```

<img src="figs/plan_erot_bcd_reg0_1.png" width="45%" /> <img src="figs/plan_erot_ssns_reg0_01.png" width="45%" />
<img src="figs/plan_qrot_pdip_reg0_1.png" width="45%" /> <img src="figs/plan_qrot_pdip_reg0_01.png" width="45%" />

🌟 **Fun fact**: The logo sticker of **RegOT** also uses the package itself to compute the transport pattern between point clouds. You can use [this code](figs/sticker.py) to reproduce the image.

![RegOT sticker](https://statr.me/images/sticker-regot.png)

### 📃 Bibliography

Please consider to cite our work if you find our algorithms or
software useful in your research and applications.

```bibtex
@inproceedings{tang2024safe,
  title={Safe and sparse Newton method for entropic-regularized optimal transport},
  author={Tang, Zihao and Qiu, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  pages={129914--129943},
  year={2024}
}

@inproceedings{wang2025sparse,
  title={The Sparse-Plus-Low-Rank quasi-Newton method for entropic-regularized optimal transport},
  author={Wang, Chenrui and Qiu, Yixuan},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
