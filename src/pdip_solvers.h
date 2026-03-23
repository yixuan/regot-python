#ifndef REGOT_PDIP_SOLVERS_H
#define REGOT_PDIP_SOLVERS_H

#include <iostream>
#include <Eigen/Core>
#include "pdip_result.h"

namespace PDIP {

using RefConstVec = Eigen::Ref<const Eigen::VectorXd>;
using RefConstMat = Eigen::Ref<const Eigen::MatrixXd>;

// Tunable PDIP parameters:
// - Defaults match the first version's usable ranges;
// - Affect inner solve paths only; public result fields keep the same meaning.
struct PDIPSolverOpts
{
    int cg_max_iter;         // Max inner CG iterations
    double fixed_threshold;  // FP sparsity threshold (gates B2 path)
    int fp_max_iter;         // Max inner fixed-point iterations for FP
    double fp_exit_scale;    // Inner FP early-exit threshold scale
    // true (default): stop only when normalized primal_gap, dual_gap, and mu are all < tol.
    // false: also allow early stop when mar < tol (legacy behavior).
    bool fp_stop_gap_mu_only;
    // PDIP-CG: true = stop on gap+mu only (matches FP semantics; inexact inner PCG can be slow/unstable).
    // Default false: keep mar / stability window like Python pdip_solver_cg for better numerics.
    bool cg_stop_gap_mu_only;
    // Marginal-error threshold for by_mar_tol / by_stable (decoupled from outer tol; default 1e-10)
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
