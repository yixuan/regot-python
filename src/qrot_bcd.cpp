#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "timer.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

void qrot_bcd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg, 0.0);
    Vector gamma(n + m), grad(n + m);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    if (opts.x0.size() == gamma.size())
    {
        gamma.noalias() = opts.x0;
    } else {
        gamma.head(n).setZero();
        prob.optimal_beta(gamma.head(n), gamma.tail(m));
    }

    // Start timing
    Timer timer;
    timer.tic();
    // Initial objective function and gradient
    double f = prob.dual_obj_grad(gamma, grad);
    double gnorm = grad.norm();
    // Record timing
    double duration = timer.toc("iter");

    // Collect progress statistics
    double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm / reg);
    run_times.push_back(duration);

    int i;
    for (i = 0; i < max_iter; i++)
    {
        if (verbose >= 1)
        {
            cout << "iter = " << i << ", objval = " << f <<
                ", ||grad|| = " << gnorm << std::endl;
        }

        // Start timing
        timer.tic();

        // Convergence test
        if (gnorm < tol)
            break;

        // Optimal alpha given beta
        prob.optimal_alpha(gamma.tail(m), gamma.head(n));
        // Optimal beta given alpha
        prob.optimal_beta(gamma.head(n), gamma.tail(m));

        // Get the new f and g
        f = prob.dual_obj_grad(gamma, grad);
        gnorm = grad.norm();
        // Record timing
        duration = timer.toc("iter");

        // Collect progress statistics
        prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm / reg);
        run_times.push_back(run_times.back() + duration);
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    result.prim_vals.swap(prim_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}

}  // namespace QROT
