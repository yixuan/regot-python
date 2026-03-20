import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import regot

# 脚本职责：
# 1) 在四个数据集上统一比较 PDIP 与 QROT；
# 2) 输出 summary.csv（含统一 primal 目标 primal_obj_unified）与 fp_profile.csv；
# 3) 生成误差/目标值随迭代与时间的可视化图。
# 默认对 reg=0.1, 0.01, 0.001 各跑一遍；图按 reg 分子目录，summary.csv 汇总全部。


def _check_regot_symbols():
    missing = [name for name in ("pdip_cg", "pdip_fp", "qrot_lbfgs_dual", "qrot_grssn") if not hasattr(regot, name)]
    if missing:
        regot_file = getattr(regot, "__file__", "unknown")
        py = sys.executable
        raise RuntimeError(
            "当前环境的 regot 缺少接口: %s\n"
            "python: %s\n"
            "regot: %s\n"
            "请在本仓库根目录重新安装当前环境中的 regot：\n"
            "  python3 -m pip install --break-system-packages -e .\n"
            "或确认你运行脚本使用的是同一个 python 解释器。"
            % (missing, py, regot_file)
        )


def _import_project_datasets(project_root: Path):
    # Reuse existing dataset definitions from sibling project folder.
    parent = project_root.parent
    sys.path.insert(0, str(parent))
    from ot.datasets import Synthetic1OT, Synthetic2OT, MnistOT, FashionMnistOT
    return Synthetic1OT, Synthetic2OT, MnistOT, FashionMnistOT


def _run_solver(name, fn, M, a, b, reg, tol, max_iter, pdip_extra_kwargs=None):
    t0 = time.time()
    kw = dict(tol=tol, max_iter=max_iter, verbose=0)
    if pdip_extra_kwargs:
        kw.update(pdip_extra_kwargs)
    res = fn(M, a, b, reg, **kw)
    wall = time.time() - t0
    plan = getattr(res, "plan", None)
    mar_errs = list(getattr(res, "mar_errs", []))
    run_times = list(getattr(res, "run_times", []))
    obj_vals = list(getattr(res, "obj_vals", []))
    final_obj = float(obj_vals[-1]) if obj_vals else float("nan")
    # 统一口径 primal：<C, Pi> + reg/2 * ||Pi||_F^2（与 C++ obj_vals 一致）
    primal_obj_unified = float("nan")
    if plan is not None:
        plan_arr = np.asarray(plan, dtype=np.float64)
        if plan_arr.size > 0 and np.isfinite(plan_arr).all():
            primal_quad = 0.5 * reg * float(np.sum(plan_arr * plan_arr))
            primal_linear = float(np.sum(M * plan_arr))
            pu = primal_linear + primal_quad
            if np.isfinite(pu):
                primal_obj_unified = pu
    if not np.isfinite(primal_obj_unified) and np.isfinite(final_obj):
        primal_obj_unified = final_obj
    fp_profile = None
    if name == "PDIP-FP":
        fp_profile = {
            "fp_t_build_B": float(getattr(res, "t_build_B", 0.0)),
            "fp_t_chol_factor": float(getattr(res, "t_chol_factor", 0.0)),
            "fp_t_chol_solve": float(getattr(res, "t_chol_solve", 0.0)),
            "fp_t_eq_matvec": float(getattr(res, "t_eq_matvec", 0.0)),
            "fp_t_other": float(getattr(res, "t_other", 0.0)),
        }
    return {
        "solver": name,
        "niter": int(getattr(res, "niter", len(mar_errs))),
        "converged": bool(getattr(res, "converged", len(mar_errs) > 0 and mar_errs[-1] < tol)),
        "final_mar_err": float(mar_errs[-1]) if mar_errs else float("nan"),
        "primal_obj_unified": primal_obj_unified,
        "time_sec": float(run_times[-1] / 1000.0) if run_times else wall,
        "mar_errs": mar_errs,
        "run_times": run_times,
        "obj_vals": obj_vals,
        "fp_profile": fp_profile,
    }


def _plot_curves(out_dir: Path, dataset_name: str, records, plot_obj: bool):
    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(111)
    for r in records:
        if not r["mar_errs"]:
            continue
        x = np.arange(1, len(r["mar_errs"]) + 1)
        y = np.maximum(np.asarray(r["mar_errs"], dtype=float), 1e-20)
        ax1.semilogy(x, y, label=r["solver"])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Marginal Error")
    ax1.set_title(f"{dataset_name}: Error vs Iteration")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{dataset_name}_err_iter.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 5))
    ax2 = fig2.add_subplot(111)
    for r in records:
        if not r["mar_errs"] or not r["run_times"] or len(r["mar_errs"]) != len(r["run_times"]):
            continue
        x = np.asarray(r["run_times"], dtype=float) / 1000.0
        y = np.maximum(np.asarray(r["mar_errs"], dtype=float), 1e-20)
        ax2.semilogy(x, y, label=r["solver"])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Marginal Error")
    ax2.set_title(f"{dataset_name}: Error vs Time")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{dataset_name}_err_time.png", dpi=150)
    plt.close(fig2)

    if plot_obj:
        fig3 = plt.figure(figsize=(8, 5))
        ax3 = fig3.add_subplot(111)
        for r in records:
            if not r["obj_vals"]:
                continue
            x = np.arange(1, len(r["obj_vals"]) + 1)
            y = np.asarray(r["obj_vals"], dtype=float)
            ax3.plot(x, y, label=r["solver"])
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Objective")
        ax3.set_title(f"{dataset_name}: Objective vs Iteration")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{dataset_name}_obj_iter.png", dpi=150)
        plt.close(fig3)

        fig4 = plt.figure(figsize=(8, 5))
        ax4 = fig4.add_subplot(111)
        for r in records:
            if not r["obj_vals"] or not r["run_times"] or len(r["obj_vals"]) != len(r["run_times"]):
                continue
            x = np.asarray(r["run_times"], dtype=float) / 1000.0
            y = np.asarray(r["obj_vals"], dtype=float)
            ax4.plot(x, y, label=r["solver"])
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Objective")
        ax4.set_title(f"{dataset_name}: Objective vs Time")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend()
        fig4.tight_layout()
        fig4.savefig(out_dir / f"{dataset_name}_obj_time.png", dpi=150)
        plt.close(fig4)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PDIP-CG/FP with QROT-LBFGS/GRSSN on four datasets."
    )
    parser.add_argument(
        "--reg",
        type=float,
        nargs="*",
        default=None,
        metavar="R",
        help="正则化系数，可多项；省略时默认 0.1 0.01 0.001。例：--reg 0.05 或 --reg 0.2 0.1",
    )
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n", type=int, default=1000, help="n=m for Synthetic I/II")
    parser.add_argument("--out-dir", type=str, default="benchmarks/results_compare_pdip_qrot_4datasets")
    parser.add_argument("--plot-obj", action="store_true", default=True, help="plot objective curves")
    parser.add_argument("--diagnose-fp", action="store_true", default=False, help="export FP phase timing profile")
    parser.add_argument(
        "--fp-allow-mar-stop",
        action="store_true",
        default=False,
        help="PDIP-FP：恢复 mar < tol 提前停机（旧行为）。默认仅用 rp∧rd∧μ。",
    )
    parser.add_argument(
        "--cg-gap-mu-only",
        action="store_true",
        default=False,
        help="PDIP-CG：仅用 rp∧rd∧μ 停机（与 FP 一致；内层 PCG 非精确时可能变慢或数值不稳）。默认 CG 仍可用 mar/稳定判据。",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_regot_symbols()

    reg_list = list(args.reg) if args.reg else [0.1, 0.01, 0.001]

    Synthetic1OT, Synthetic2OT, MnistOT, FashionMnistOT = _import_project_datasets(root)
    solvers = [
        ("PDIP-CG", regot.pdip_cg),
        ("PDIP-FP", regot.pdip_fp),
        ("QROT-LBFGS", regot.qrot_lbfgs_dual),
        ("QROT-GRSSN", regot.qrot_grssn),
    ]

    rows = []
    fp_profile_rows = []
    for reg_val in reg_list:
        problems = [
            ("SyntheticI", Synthetic1OT(n=args.n, m=args.n, reg=reg_val)),
            ("SyntheticII", Synthetic2OT(n=args.n, m=args.n, reg=reg_val)),
            ("MNIST", MnistOT(reg=reg_val)),
            ("FashionMNIST", FashionMnistOT(reg=reg_val)),
        ]
        reg_plot_dir = out_dir / f"reg_{reg_val:g}"
        reg_plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n======== reg = {reg_val:g}  (plots -> {reg_plot_dir.relative_to(root)}) ========")

        for dataset_name, prob in problems:
            M = np.asarray(prob.M, dtype=np.float64)
            a = np.asarray(prob.a, dtype=np.float64).ravel()
            b = np.asarray(prob.b, dtype=np.float64).ravel()
            a = a / max(a.sum(), 1e-10)
            b = b / max(b.sum(), 1e-10)
            print(f"\n[{dataset_name}]  reg={reg_val:g}  n={len(a)}  m={len(b)}")
            print(
                f"  {'solver':<14} {'time(s)':>9} {'niter':>7} {'mar':>12} "
                f"{'primal':>14} {'conv':>5}"
            )
            print(
                f"  {'-' * 14} {'-' * 9} {'-' * 7} {'-' * 12} "
                f"{'-' * 14} {'-' * 5}"
            )

            records = []
            for sname, sfn in solvers:
                # PDIP-CG / PDIP-FP：均用 --tol（默认 1e-8），默认仅 rp∧rd∧μ；QROT：--tol。
                extra = None
                if sname == "PDIP-FP":
                    tol_used = args.tol
                    if args.fp_allow_mar_stop:
                        extra = {"fp_stop_gap_mu_only": False}
                elif sname == "PDIP-CG":
                    tol_used = args.tol
                    if args.cg_gap_mu_only:
                        extra = {"cg_stop_gap_mu_only": True}
                else:
                    tol_used = args.tol
                rec = _run_solver(
                    sname, sfn, M, a, b, reg_val, tol_used, args.max_iter,
                    pdip_extra_kwargs=extra,
                )
                conv_s = "yes" if rec["converged"] else "no"
                po = rec["primal_obj_unified"]
                primal_s = f"{po:14.6e}" if np.isfinite(po) else f"{'nan':>14}"
                print(
                    f"  {sname:<14} {rec['time_sec']:9.3f} {rec['niter']:7d} "
                    f"{rec['final_mar_err']:12.3e} {primal_s} {conv_s:>5}"
                )
                rows.append({
                    "dataset": dataset_name,
                    "n": len(a),
                    "m": len(b),
                    "solver": rec["solver"],
                    "time_sec": rec["time_sec"],
                    "niter": rec["niter"],
                    "final_mar_err": rec["final_mar_err"],
                    "primal_obj_unified": rec["primal_obj_unified"],
                    "converged": rec["converged"],
                    "reg": reg_val,
                    "tol": tol_used,
                    "max_iter": args.max_iter,
                })
                if args.diagnose_fp and rec["fp_profile"] is not None:
                    fp_row = {"dataset": dataset_name, "solver": rec["solver"], "reg": reg_val}
                    fp_row.update(rec["fp_profile"])
                    fp_profile_rows.append(fp_row)
                records.append(rec)
            _plot_curves(reg_plot_dir, dataset_name, records, args.plot_obj)

    csv_path = out_dir / "summary.csv"
    # summary.csv：汇总全部 --reg 取值；primal_obj_unified = <C,Pi> + reg/2||Pi||_F^2。
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "n", "m", "solver", "time_sec",
                "niter", "final_mar_err", "primal_obj_unified",
                "converged", "reg", "tol", "max_iter",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {csv_path}")

    if args.diagnose_fp and fp_profile_rows:
        fp_csv = out_dir / "fp_profile.csv"
        # fp_profile.csv 读取建议：
        # 先看 fp_t_chol_factor 占比判断分解是否为主瓶颈；
        # 再看 fp_t_other 评估循环内向量运算与更新成本。
        with fp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset", "solver", "reg",
                    "fp_t_build_B", "fp_t_chol_factor", "fp_t_chol_solve",
                    "fp_t_eq_matvec", "fp_t_other",
                ],
            )
            writer.writeheader()
            writer.writerows(fp_profile_rows)
        print(f"Saved FP profile: {fp_csv}")


if __name__ == "__main__":
    main()
