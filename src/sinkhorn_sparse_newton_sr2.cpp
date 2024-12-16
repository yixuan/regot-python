#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "sinkhorn_hess.h"
#include "sinkhorn_linsolve.h"
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void sinkhorn_sparse_newton_sr2_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{

}

}  // namespace Sinkhorn
