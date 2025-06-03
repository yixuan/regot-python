#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "timer.h"
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

void sinkhorn_bcd_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg);
    Vector alpha(n), beta(m), gamma(n + m - 1), grad(n + m - 1);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    if (opts.x0.size() == gamma.size())
    {
        alpha.noalias() = opts.x0.head(n);
        beta.head(m - 1).noalias() = opts.x0.tail(m - 1);
        beta[m - 1] = 0.0;
    } else {
        alpha.setZero();
        prob.optimal_beta(alpha, beta);
    }

    // Start timing
    Timer timer, timer_inner;
    timer.tic();
    // Initial objective function and gradient
    gamma.head(n).array() = alpha.array() + beta[m - 1];
    gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];
    double f = prob.dual_obj_grad(gamma, grad);
    double gnorm = grad.norm();
    // Record timing
    double duration = timer.toc("iter");

    // Collect progress statistics
    // double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    // prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm);
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
        timer_inner.tic();

        // Convergence test
        if (gnorm < tol)
            break;

        // Optimal alpha given beta
        prob.optimal_alpha(beta, alpha);
        timer_inner.toc("alpha");

        // Optimal beta given alpha
        prob.optimal_beta(alpha, beta);
        timer_inner.toc("beta");

        gamma.head(n).array() = alpha.array() + beta[m - 1];
        gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];

        // Get the new f and g
        f = prob.dual_obj_grad(gamma, grad);
        gnorm = grad.norm();
        // Record timing
        timer_inner.toc("grad");
        duration = timer.toc("iter");

        if (verbose >= 2)
        {
            cout << "[timing]=================================================" << std::endl;
            cout << "║ alpha = " << timer_inner["alpha"] <<
                ", beta = " << timer_inner["beta"] <<
                ", grad = " << timer_inner["grad"] << std::endl;
            cout << "=========================================================" << std::endl << std::endl;
        }

        // Collect progress statistics
        // prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        // prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm);
        run_times.push_back(run_times.back() + duration);
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    // result.prim_vals.swap(prim_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}

}  // namespace Sinkhorn
