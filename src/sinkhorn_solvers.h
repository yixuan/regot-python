#ifndef REGOT_SINKHORN_SOLVERS_H
#define REGOT_SINKHORN_SOLVERS_H

#include <iostream>
#include <Eigen/Core>
#include "sinkhorn_result.h"

namespace Sinkhorn {

using RefConstVec = Eigen::Ref<const Eigen::VectorXd>;
using RefConstMat = Eigen::Ref<const Eigen::MatrixXd>;

// Extra options that may be used by the solvers
struct SinkhornSolverOpts
{
    // Initial value
    Eigen::VectorXd x0;
    // Method for solving linear systems
    int method;
    // Parameter for SSNS
    double mu0;
    // Parameter for sparse Newton
    double shift;
    double density;

    // Setting default values
    SinkhornSolverOpts():
        x0(0), method(0), mu0(1.0), shift(1e-6), density(0.01)
    {}
};

void sinkhorn_bcd_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void sinkhorn_apdagd_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void sinkhorn_lbfgs_dual_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void sinkhorn_newton_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void sinkhorn_sparse_newton_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void sinkhorn_sparse_newton_low_rank_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

void sinkhorn_ssns_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, int verbose = 0,
    std::ostream& cout = std::cout
);

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_SOLVERS_H
