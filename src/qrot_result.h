#ifndef REGOT_QROT_RESULT_H
#define REGOT_QROT_RESULT_H

#include <vector>
#include "qrot_problem.h"
#include "config.h"

struct QROTResult
{
// Determine matrix type according to configuration
#ifdef REGOT_USE_ROW_MAJOR_MATRIX
    using ResultMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
    using ResultMatrix = Eigen::MatrixXd;
#endif

    using Vector = Eigen::VectorXd;

    int                 niter;
    ResultMatrix        plan;
    std::vector<double> obj_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Recover the transport plan given the dual variables
    inline void get_plan(const Vector& gamma, const Problem& prob)
    {
        // alpha = gamma[:n], beta = gamma[n:]
        const int n = prob.size_n();
        const int m = prob.size_m();
        const double reg = prob.reg();

        this->plan.resize(n, m);
        this->plan.noalias() = gamma.head(n).replicate(1, m) +
            gamma.tail(m).transpose().replicate(n, 1) -
            prob.get_M();
        this->plan.noalias() = this->plan.cwiseMax(0.0) / reg;
    }
};

#endif  // REGOT_QROT_RESULT_H
