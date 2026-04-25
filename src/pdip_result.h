#ifndef REGOT_PDIP_RESULT_H
#define REGOT_PDIP_RESULT_H

#include <vector>
#include <Eigen/Core>
#include "config.h"

namespace PDIP {

// PDIP result struct (same style as other regot solvers):
// - Primary fields for plotting and summaries in user scripts;
// - Diagnostic fields for performance analysis only, not for convergence/correctness.
struct PDIPResult
{
#ifdef REGOT_USE_ROW_MAJOR_MATRIX
    using ResultMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
    using ResultMatrix = Eigen::MatrixXd;
#endif

    int                 niter = 0;         // Outer iteration count
    bool                converged = false; // Whether termination criteria were met
    ResultMatrix        plan;              // Transport plan, shape (n, m)
    std::vector<double> obj_vals;          // Objective value per outer iteration
    std::vector<double> mar_errs;          // Marginal error per outer iteration
    std::vector<double> run_times;         // Cumulative wall time per outer iteration (ms)
    // Optional PDIP-FP per-phase profiling times (seconds)
    double              t_build_B = 0.0;
    double              t_chol_factor = 0.0;
    double              t_chol_solve = 0.0;
    double              t_eq_matvec = 0.0;
    double              t_other = 0.0;
};

}  // namespace PDIP

#endif  // REGOT_PDIP_RESULT_H
