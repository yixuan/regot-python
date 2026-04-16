#!/usr/bin/env python3
"""
Regenerate README transport-plan PNGs:
  EROT: `plan_erot_bcd_reg0_1.png`, `plan_erot_ssns_reg0_01.png`
  QROT PDIP: `plan_pdip_reg0_1.png`, `plan_pdip_reg0_01.png`
Does not overwrite `figs/plan_reg0_*.png`.

Run from repository root (recommended):
    python figs/generate_readme_plans.py

Or from this directory:
    python generate_readme_plans.py
"""
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, norm

import regot

FIGS = Path(__file__).resolve().parent
REPO_ROOT = FIGS.parent


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


def vis_plan(T, title: str = "", cmap: str = "viridis", save_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(T, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=20)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> int:
    os.chdir(REPO_ROOT)
    np.random.seed(123)
    M, a, b = example(n=100, m=80)

    reg = 0.1
    res1 = regot.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
    reg = 0.01
    res2 = regot.sinkhorn_ssns(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)

    res3 = regot.qrot_pdip(M, a, b, 0.1, max_iter=2000, inner_solver="cg")
    reg = 0.01
    res4 = regot.qrot_pdip(M, a, b, reg, max_iter=2000, inner_solver="fp")

    vis_plan(res1.plan, title="Sinkhorn (BCD), reg=0.1", save_path=FIGS / "plan_erot_bcd_reg0_1.png")
    vis_plan(res2.plan, title="SSNS, reg=0.01", save_path=FIGS / "plan_erot_ssns_reg0_01.png")
    vis_plan(res3.plan, title="reg=0.1", save_path=FIGS / "plan_pdip_reg0_1.png")
    vis_plan(res4.plan, title="reg=0.01", save_path=FIGS / "plan_pdip_reg0_01.png")

    for name in (
        "plan_erot_bcd_reg0_1.png",
        "plan_erot_ssns_reg0_01.png",
        "plan_pdip_reg0_1.png",
        "plan_pdip_reg0_01.png",
    ):
        print("Wrote:", FIGS / name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print("ImportError:", e, file=sys.stderr)
        print("Install the package first, e.g. pip install -e .", file=sys.stderr)
        raise SystemExit(1)
