#ifndef REGOT_SINKHORN_RESULT_H
#define REGOT_SINKHORN_RESULT_H

#include <vector>
#include "config.h"
#include "sinkhorn_problem.h"

namespace Sinkhorn {

struct SinkhornResult
{
// Determine matrix type according to configuration
#ifdef REGOT_USE_ROW_MAJOR_MATRIX
    using ResultMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
    using ResultMatrix = Eigen::MatrixXd;
#endif

    using Vector = Eigen::VectorXd;

    int                 niter;
    Vector              dual;
    ResultMatrix        plan;
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;
    std::vector<double> density;

    // Recover the transport plan given the dual variables
    inline void get_plan(const Vector& gamma, const Problem& prob)
    {
        // alpha = gamma[:n], betat = gamma[n:]
        const int n = prob.size_n();
        const int m = prob.size_m();
        const double reg = prob.reg();

        // Extract betat and set beta=(betat, 0)
        Vector beta(m);
        beta.head(m - 1).noalias() = gamma.tail(m - 1);
        beta[m - 1] = 0.0;

        // Compute logT = (alpha (+) beta - M) / reg
        this->plan.resize(n, m);
        this->plan.noalias() = (gamma.head(n).replicate(1, m) +
            beta.transpose().replicate(n, 1) - prob.get_M()) / reg;
        // Compute T
        this->plan.array() = this->plan.array().exp();
    }
};

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_RESULT_H
