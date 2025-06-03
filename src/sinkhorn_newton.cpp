#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "timer.h"
#include "sinkhorn_hess.h"
#include "sinkhorn_linsolve.h"
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

void sinkhorn_newton_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double theta = 0.5, kappa = 0.5;
    constexpr int nlinesearch = 20;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m - 1), newgamma(n + m - 1), direc(n + m - 1);

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
        Vector beta(m);
        prob.optimal_beta(gamma.head(n), beta);
        gamma.head(n).array() += beta[m - 1];
        gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];
    }

    // Start timing
    Timer timer;
    timer.tic();
    // Initial objective function value, gradient, and Hessian
    double f;
    Vector g;
    Matrix H;
    prob.dual_obj_grad_densehess(gamma, f, g, H);
    double gnorm = g.norm();
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

        // Convergence test
        // Also exit if objective function value is not finite
        if ((gnorm < tol) || (!std::isfinite(f)))
            break;

        // Compute search direction
        Eigen::LLT<Matrix> linsolver(H);
        direc.noalias() = linsolver.solve(-g);

        // Line search
        double step = 1.0;
        double thresh = theta * g.dot(direc);
        for (int k = 0; k < nlinesearch; k++)
        {
            newgamma.noalias() = gamma + step * direc;
            const double newf = prob.dual_obj(newgamma);
            if (newf <= f + step * thresh)
                break;
            step *= kappa;
        }
        gamma.swap(newgamma);

        // Get the new f, g, H
        prob.dual_obj_grad_densehess(gamma, f, g, H);
        gnorm = g.norm();
        // Record timing
        duration = timer.toc("iter");

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
