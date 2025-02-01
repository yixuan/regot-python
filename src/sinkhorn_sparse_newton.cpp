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
    int method = opts.method;
    constexpr double cg_tol = 1e-8;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m - 1), newgamma(n + m - 1), direc(n + m - 1);
    double gnorm;

    // Progress statistics
    std::vector<double> obj_vals;
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

    // Linear solver
    SinkhornLinearSolver lin_sol;
    lin_sol.method = method;
    lin_sol.cg_tol = cg_tol;
    lin_sol.verbose = verbose;

    // Start timing
    TimePoint clock_t1 = Clock::now();
    // Initial objective function value, gradient, and Hessian
    double f;
    Vector g;
    Hessian H;
    Matrix T(n, m);
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
        clock_t1 = Clock::now();

        // Convergence test
        // Also exit if objective function value is not finite
        if ((gnorm < tol) || (!std::isfinite(f)))
            break;

        // Compute search direction
        const double shift = std::min(gnorm, shift_max);
        TimePoint clock_s1 = Clock::now();
        lin_sol.solve(direc, H, -g, shift);
        TimePoint clock_s2 = Clock::now();

        // Armijo Line Search
        alpha = prob.line_search_wolfe(
            std::min(1.0, 1.5 * alpha), gamma, direc, f, g, T
        );
        gamma.noalias() += alpha * direc;
        TimePoint clock_s3 = Clock::now();

        // Get the new f, g, H
        // T has been computed in line search
        f = prob.dual_obj_grad(gamma, g, T, false);
        TimePoint clock_s4 = Clock::now();
        // Adjust density according to gnorm change
        const double gnorm_pre = gnorm;
        gnorm = g.norm();
        density *= (gnorm < gnorm_pre) ? 0.9 : 2.0;
        density = std::min(density_max, std::max(density_min, density));
        // Compute H
        prob.dual_sparsified_hess_with_density(T, g, density, H);
        TimePoint clock_s5 = Clock::now();

        if (verbose >= 2)
        {
            cout << "[timing]---------------------------------------------------" << std::endl;
            cout << "║ lin_solve = " << (clock_s2 - clock_s1).count() <<
                ", line_search = " << (clock_s3 - clock_s2).count() << std::endl;
            cout << "║ grad = " << (clock_s4 - clock_s3).count() <<
                ", hess = " << (clock_s5 - clock_s4).count() << std::endl;
            cout << "===========================================================" << std::endl << std::endl;
        }

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
