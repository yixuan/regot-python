#ifndef REGOT_PDIP_SOLVERS_H
#define REGOT_PDIP_SOLVERS_H

#include <iostream>
#include <Eigen/Core>
#include "pdip_result.h"

namespace PDIP {

using RefConstVec = Eigen::Ref<const Eigen::VectorXd>;
using RefConstMat = Eigen::Ref<const Eigen::MatrixXd>;

// PDIP 算法可调参数：
// - 默认值保持第一版可用区间；
// - 只影响内部求解路径，不改变对外结果字段语义。
struct PDIPSolverOpts
{
    int cg_max_iter;         // CG 内层最大迭代次数
    double fixed_threshold;  // FP 稀疏阈值（用于 B2 路径判定）
    int fp_max_iter;         // FP 内层固定点迭代上限
    double fp_exit_scale;    // FP 内层提前退出阈值比例
    // true（默认）：仅当归一化 primal_gap、dual_gap 与 μ 均 < tol 时停机。
    // 设为 false 时额外允许 mar < tol 提前停机（旧行为）。
    bool fp_stop_gap_mu_only;
    // CG 版 PDIP：true = 仅 gap+μ 停机（与 FP 语义一致，但内层 PCG 非精确时易慢/不稳）。
    // 默认 false：保留 mar / 稳定窗，与 Python pdip_solver_cg 行为一致、数值更稳。
    bool cg_stop_gap_mu_only;
    // 用于 by_mar_tol / by_stable 的边际误差阈值（与外层 tol 解耦，默认 1e-10）
    double cg_mar_tol;

    PDIPSolverOpts():
        cg_max_iter(1000), fixed_threshold(1e-9), fp_max_iter(800), fp_exit_scale(1e-2),
        fp_stop_gap_mu_only(true), cg_stop_gap_mu_only(false), cg_mar_tol(1e-10)
    {}
};

void pdip_cg_internal(
    PDIPResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const PDIPSolverOpts& opts,
    double tol = 1e-8, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void pdip_fp_internal(
    PDIPResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const PDIPSolverOpts& opts,
    double tol = 1e-8, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

}  // namespace PDIP

#endif  // REGOT_PDIP_SOLVERS_H
