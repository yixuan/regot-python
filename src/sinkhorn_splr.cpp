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

void sinkhorn_splr_internal(
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
    constexpr double eps = 1e-6;  // std::numeric_limits<double>::epsilon();

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m - 1), gamma_pre(n + m - 1), direc(n + m - 1);
    double gnorm, gnorm_init;
    Vector y(n + m - 1), s(n + m - 1);

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
    gamma_pre.setZero();

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
    Vector g(n + m - 1), g_pre(n + m - 1);
    Hessian H;
    Matrix T(n, m);
    // We do not do low-rank update in the first iteration,
    // so we do not need to compute g_pre, as it will be overwritten later
    g_pre.setZero();
    // Compute f, g, T
    f = prob.dual_obj_grad(gamma, g, T, true);
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

        // Compute y and s
        y.noalias() = g - g_pre;
        s.noalias() = gamma - gamma_pre;

        // Compute search direction
        // We do not do low-rank update in the first iteration
        // When <y, s> is too small, don't use low-rank update
        const double ys = y.dot(s);
        const double yy = y.squaredNorm();
        const bool low_rank = (i > 0) && (ys > (eps * yy));
        const double shift = std::min(gnorm, shift_max);
        if (low_rank) {
            lin_sol.solve_low_rank(direc, H, -g, shift, y, s);
        } else {
            lin_sol.solve(direc, H, -g, shift);
        }
        timer_inner.toc("lin_solve");

        // Wolfe Line Search
        bool recompute_T = true;
        alpha = prob.line_search_wolfe(
            std::min(1.0, 1.5 * alpha), gamma, direc, f, g, T, recompute_T
        );
        // Save gamma to gamma_pre
        gamma_pre.noalias() = gamma;
        // Update gamma
        gamma.noalias() += alpha * direc;
        timer_inner.toc("line_search");

        // Get the new f, g, H
        // Save g to g_pre
        g_pre.swap(g);
        // Compute f and g
        // Typically, T has been computed in line search
        f = prob.dual_obj_grad(gamma, g, T, recompute_T);
        timer_inner.toc("grad");
        // Adjust density according to gnorm change
        const double gnorm_pre = gnorm;
        gnorm = g.norm();
        const bool bad_move = (gnorm_pre < gnorm_init) && (gnorm > gnorm_pre);
        density *= (bad_move ? 1.1 : 0.99);
        density = std::min(density_max, std::max(density_min, density));
        // Compute H
        prob.dual_sparsified_hess_with_density(T, g, density, H);
        timer_inner.toc("hess");

        if (verbose >= 2)
        {
            cout << "[lowrank]---------------------------------------------------" << std::endl;
            cout << "║ ys = " << ys << ", yy = " << yy << std::endl;
            cout << "║ low_rank = " << low_rank << std::endl;
            cout << "===========================================================" << std::endl;
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
