#ifndef REGOT_PDIP_RESULT_H
#define REGOT_PDIP_RESULT_H

#include <vector>
#include "config.h"
#include <Eigen/Core>

namespace PDIP {

// PDIP 对外结果容器（与 regot 其他算法保持统一风格）：
// - 主字段用于上层脚本统一绘图与汇总；
// - 诊断字段仅用于性能分析，不参与收敛/正确性判断。
struct PDIPResult
{
#ifdef REGOT_USE_ROW_MAJOR_MATRIX
    using ResultMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
    using ResultMatrix = Eigen::MatrixXd;
#endif

    int                 niter = 0;         // 外层迭代次数
    bool                converged = false; // 是否满足终止条件
    ResultMatrix        plan;              // 运输计划，shape=(n,m)
    std::vector<double> obj_vals;          // 每次外迭代目标值
    std::vector<double> mar_errs;          // 每次外迭代边际误差
    std::vector<double> run_times;         // 每次外迭代累计耗时（毫秒）
    // 以下为 PDIP-FP 可选分段诊断耗时（单位：秒）
    double              t_build_B = 0.0;
    double              t_chol_factor = 0.0;
    double              t_chol_solve = 0.0;
    double              t_eq_matvec = 0.0;
    double              t_other = 0.0;
};

}  // namespace PDIP

#endif  // REGOT_PDIP_RESULT_H
