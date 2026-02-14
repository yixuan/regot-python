#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "timer.h"
#include "sinkhorn_hess.h"
#include "sinkhorn_linsolve.h"
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Convert density of T to density of H
inline double density_T2H(double denT, int n, int m)
{
    const double nnz = 2.0 * denT * n * (m - 1) + n + m - 1;
    const double denH = nnz / double(n + m - 1) / double(n + m - 1);
    return denH;
}

void sinkhorn_sparse_newton_internal(
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
    const double density_max = opts.density,
        density_min = 0.01 * opts.density;
    double density = 0.1 * density_max;
    const double shift_max = opts.shift;
    const int method = opts.method;
    constexpr double cg_tol = 1e-8;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m - 1), newgamma(n + m - 1), direc(n + m - 1);
    double gnorm, gnorm_init;

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;
    std::vector<double> densities;

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

    // Linear solver
    SinkhornLinearSolver lin_sol;
    lin_sol.method = method;
    lin_sol.cg_tol = cg_tol;
    lin_sol.verbose = verbose;

    // Start timing
    Timer timer, timer_inner;
    timer.tic();
    // Initial objective function value, gradient, and Hessian
    double f;
    Vector g;
    Hessian H;
    Matrix T(n, m);
    f = prob.dual_obj_grad(gamma, g, T, true); // compute f, g, T
    gnorm = g.norm();
    gnorm_init = gnorm;
    // Compute H
    prob.dual_sparsified_hess_with_density(T, g, density, H);
    // Record timing
    double duration = timer.toc("iter");

    // Collect progress statistics
    obj_vals.push_back(f);
    mar_errs.push_back(gnorm);
    run_times.push_back(duration);
    densities.push_back(density_T2H(density, n, m));

    int i;
    // Initial step size
    double alpha = 1.0;
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
        // Also exit if objective function value is not finite
        if ((gnorm < tol) || (!std::isfinite(f)))
            break;

        // Compute search direction
        const double shift = std::min(gnorm, shift_max);
        lin_sol.solve(direc, H, -g, shift);
        timer_inner.toc("lin_solve");

        // Wolfe Line Search
        bool recompute_T = true;
        alpha = prob.line_search_wolfe(
            std::min(1.0, 1.5 * alpha), gamma, direc, f, g, T, recompute_T
        );
        // Update gamma
        gamma.noalias() += alpha * direc;
        timer_inner.toc("line_search");

        // Get the new f, g, H
        // Typically, T has been computed in line search
        f = prob.dual_obj_grad(gamma, g, T, recompute_T);
        timer_inner.toc("grad");
        // Adjust density according to gnorm change
        const double gnorm_pre = gnorm;
        gnorm = g.norm();
        const bool bad_move = (gnorm_pre < gnorm_init) && (gnorm > 1.1 * gnorm_pre);
        density *= (bad_move ? 2.0 : 0.9);
        density = std::min(density_max, std::max(density_min, density));
        // Compute H
        prob.dual_sparsified_hess_with_density(T, g, density, H);
        timer_inner.toc("hess");

        if (verbose >= 2)
        {
            cout << "[timing]---------------------------------------------------" << std::endl;
            cout << "║ lin_solve = " << timer_inner["lin_solve"] <<
                ", line_search = " << timer_inner["line_search"] << std::endl;
            cout << "║ grad = " << timer_inner["grad"] <<
                ", hess = " << timer_inner["hess"] << std::endl;
            cout << "===========================================================" << std::endl << std::endl;
        }

        // Record timing
        duration = timer.toc("iter");

        // Collect progress statistics
        obj_vals.push_back(f);
        mar_errs.push_back(gnorm);
        run_times.push_back(run_times.back() + duration);
        densities.push_back(density_T2H(density, n, m));
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
    result.densities.swap(densities);
}

}  // namespace Sinkhorn
