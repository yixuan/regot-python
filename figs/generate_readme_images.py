#!/usr/bin/env python3
"""
Generate README transport plan images:
    EROT: `plan_erot_bcd_reg0_1.png`, `plan_erot_ssns_reg0_01.png`
    QROT: `plan_qrot_pdip_reg0_1.png`, `plan_qrot_pdip_reg0_01.png`

Run code:
    python generate_readme_plans.py
"""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.stats import expon, norm
import regot
import matplotlib.pyplot as plt

def example(n: int = 100, m: int = 80):
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

def vis_plan(T, title: str = "", cmap: str = "viridis", filename: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(T, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=20)
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)

def main():
    np.random.seed(123)
    M, a, b = example(n=100, m=80)

    res1 = regot.sinkhorn_bcd(M, a, b, reg=0.1, tol=1e-6, max_iter=1000, verbose=0)
    res2 = regot.sinkhorn_ssns(M, a, b, reg=0.01, tol=1e-6, max_iter=1000, verbose=0)

    res3 = regot.qrot_pdip(M, a, b, reg=0.1, tol=1e-8, max_iter=1000, inner_solver="cg")
    res4 = regot.qrot_pdip(M, a, b, reg=0.01, tol=1e-8, max_iter=1000, inner_solver="fp")

    vis_plan(res1.plan, title="EROT (BCD), reg=0.1", filename="plan_erot_bcd_reg0_1.png")
    vis_plan(res2.plan, title="EROT (SSNS), reg=0.01", filename="plan_erot_ssns_reg0_01.png")
    vis_plan(res3.plan, title="QROT (PDIP-CG), reg=0.1", filename="plan_qrot_pdip_reg0_1.png")
    vis_plan(res4.plan, title="QROT (PDIP-FP), reg=0.01", filename="plan_qrot_pdip_reg0_01.png")

if __name__ == "__main__":
    main()
