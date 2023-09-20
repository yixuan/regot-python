#ifndef REGOT_QROT_SOLVERS_H
#define REGOT_QROT_SOLVERS_H

#include <iostream>
#include <Eigen/Core>
#include "qrot_result.h"

using RefConstVec = Eigen::Ref<const Eigen::VectorXd>;
using RefConstMat = Eigen::Ref<const Eigen::MatrixXd>;

void qrot_apdagd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_assn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_bcd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_grssn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, double shift = 0.001,
    bool verbose = false, std::ostream& cout = std::cout
);

void qrot_lbfgs_dual_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_lbfgs_semi_dual_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_pdaam_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);

void qrot_s3n_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false,
    std::ostream& cout = std::cout
);


#endif  // REGOT_QROT_SOLVERS_H
