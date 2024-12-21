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

void sinkhorn_sparse_newton_low_rank_internal(
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
    double density = opts.density;
    double shift = opts.shift;
    int method = opts.method;
    double cg_tol = 1e-8;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m - 1), gamma_pre(n + m - 1);
    Vector y(n + m - 1), s(n + m - 1);
    double f, f_pre;
    Vector g, g_pre;
    double gnorm;
    Hessian H;
    Vector direc;
    Matrix T(n, m);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    gamma.head(n).setZero();
    Vector beta(m);
    prob.optimal_beta(gamma.head(n), beta);
    gamma.head(n).array() += beta[m - 1];
    gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];
    gamma_pre.setZero();

    // Linear solver
    SinkhornLinearSolver lin_sol;
    lin_sol.method = method;
    lin_sol.cg_tol = cg_tol;
    lin_sol.verbose = verbose;

    // Start timing
    TimePoint clock_t1 = Clock::now();
    // Initial objective function value, gradient, and Hessian
    f_pre = prob.dual_obj_grad(gamma_pre, g_pre); // compute f_pre, g_pre, T_pre
    f = prob.dual_obj_grad(gamma, g, T, true); // compute f, g, T
    gnorm = g.norm();
    prob.dual_sparsified_hess_with_density(T, g, density, H);
    // Record timing
    TimePoint clock_t2 = Clock::now();

    // Collect progress statistics
    obj_vals.push_back(f);
    mar_errs.push_back(gnorm);
    run_times.push_back((clock_t2 - clock_t1).count());

    int i;
    for (i = 0; i < max_iter; i++)
    {
        if (verbose >= 1)
        {
            cout << "iter = " << i << ", objval = " << f <<
                ", ||grad|| = " << gnorm << std::endl;
        }

        // Start timing
        clock_t1 = Clock::now();

        // Convergence test
        // Also exit if objective function value is not finite
        if ((gnorm < tol) || (!std::isfinite(f)))
            break;

        // Compute y and s
        y = g - g_pre;
        s = gamma - gamma_pre;

        // Compute search direction
        lin_sol.solve_low_rank(direc, H, -g, shift, y, s);

        // Armijo Line Search
        double alpha = prob.line_selection_armijo(
            gamma, direc, f, g
        );
        gamma_pre = gamma; // save gamma
        gamma = gamma + alpha * direc;

        // Get the new f, g, H
        f_pre = f, g_pre = g; // save f, g
        f = prob.dual_obj_grad(gamma, g, T, true); // compute f, g, T
        gnorm = g.norm();
        prob.dual_sparsified_hess_with_density(T, g, density, H);
        // Record timing
        clock_t2 = Clock::now();

        // Collect progress statistics
        obj_vals.push_back(f);
        mar_errs.push_back(gnorm);
        double duration = (clock_t2 - clock_t1).count();
        run_times.push_back(run_times.back() + duration);
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}

}  // namespace Sinkhorn

