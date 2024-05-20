#ifndef REGOT_QROT_SOLVERS_H
#define REGOT_QROT_SOLVERS_H

#include <iostream>
#include <Eigen/Core>
#include "qrot_result.h"

namespace QROT {

using RefConstVec = Eigen::Ref<const Eigen::VectorXd>;
using RefConstMat = Eigen::Ref<const Eigen::MatrixXd>;

// Extra options that may be used by the solvers
struct QROTSolverOpts
{
    // Initial value
    Eigen::VectorXd x0;
    // Extra regularization term
    double tau;
    // Shift term for regularized Newton method
    double shift;
    // Method for solving linear systems
    int method;
    // Parameter for S5N
    double mu0;

    // Setting default values
    QROTSolverOpts():
        x0(0), tau(0.0), shift(1e-3), method(0), mu0(1.0)
    {}
};

void qrot_apdagd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_assn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_bcd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_gd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_grssn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000,
    bool verbose = false, std::ostream& cout = std::cout
);

void qrot_lbfgs_dual_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_lbfgs_semi_dual_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_pdaam_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_s5n_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

}  // namespace QROT


#endif  // REGOT_QROT_SOLVERS_H
