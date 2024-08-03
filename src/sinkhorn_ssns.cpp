#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "sinkhorn_hess.h"
#include "sinkhorn_linsolve.h"
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

// Whether to print detailed timing information
// #define TIMING 1

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void sinkhorn_ssns_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Solver options
    int method = opts.method;
    double mu0 = opts.mu0;

    // Algorithmic parameters
    constexpr double rho_t1 = 0.25, rho_t2 = 0.75,
        mu_s1 = 4.0, mu_s2 = 0.5, kappa = 0.001,
        pow_lam = 1.0, pow_delta = 1.0,
        cg_tol = 1e-8, nu0 = 0.01;

    // Parameters that are adjusted in each iteration
    double mu = mu0, rho = 1.0;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m - 1), step(n + m - 1), Hstep(n + m - 1), direc(n + m - 1);
    Vector cg_x0(m - 1);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;
    std::vector<double> density;

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
    f = prob.dual_obj_grad(gamma, g, T, true);
    double gnorm = g.norm();
    double delta = nu0 * std::pow(gnorm, pow_delta);
    prob.dual_sparsified_hess(T, g, delta, 0.001, H);
    // Record timing
    TimePoint clock_t2 = Clock::now();

    // Collect progress statistics
    double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm);
    run_times.push_back((clock_t2 - clock_t1).count());
    density.push_back(H.density());

    int i;
    // std::vector<double> alphas{1.0, 0.5, 0.25, 0.125, 0.0625, 0.01};
    std::vector<double> alphas{1.0, 0.5, 0.25, 0.1};
    // std::vector<double> alphas{1.0, 0.5, 0.1};
    // std::vector<double> alphas_small{0.05, 0.01, 0.001, 1e-4};
    // bool use_alpha_small = true;
    for (i = 0; i < max_iter; i++)
    {
        if (verbose)
        {
            cout << "i = " << i << ", obj = " << f <<
                ", gnorm = " << gnorm << std::endl;
            cout << "rho = " << rho << ", mu = " << mu <<
                ", density = " << H.density() << std::endl;
        }

        // Start timing
        clock_t1 = Clock::now();

        // Convergence test
        if (gnorm < tol)
            break;

        // Compute search direction
        double shift = mu * std::pow(gnorm, pow_lam);
        // double shift = mu * std::pow(dgap, delta);
        // cout << ", shift = " << shift << std::endl;
        if (rho <= 0.0)
        {
            // No update in previous iteration
            // Use last direction for warm start
            lin_sol.cg_x0.noalias() = direc.tail(m);
        } else {
            lin_sol.cg_x0.setZero();
        }
        bool same_sparsity = (rho <= 0.0);
#ifdef TIMING
        TimePoint clock_s1 = Clock::now();
#endif
        lin_sol.solve(direc, H, -g, shift, !same_sparsity, cout);
#ifdef TIMING
        TimePoint clock_s2 = Clock::now();
#endif

        // Step size selection
        double newf;
        double alpha = prob.line_selection(
            alphas, gamma, direc, f, T, newf, verbose, cout);
        // if (newf >= f)
        // {
        //     if (use_alpha_small)
        //     {
        //         alpha = prob.line_selection(
        //             alphas_small, gamma, direc, f, newf, verbose, cout);
        //     }
        //     use_alpha_small = !use_alpha_small;
        // }
        // else
        // {
        //     use_alpha_small = true;
        // }

#ifdef TIMING
        TimePoint clock_s3 = Clock::now();
#endif

        // Estimate rho
        step.noalias() = alpha * direc;
        H.apply_Hsx(step, Hstep);
#ifdef TIMING
        TimePoint clock_s4 = Clock::now();
#endif
        const double numer = f - newf;
        const double denom = -g.dot(step) - 0.5 * step.dot(Hstep);
        rho = numer / denom;
        // std::cout << "alpha = " << alpha <<
        //     ", numer = " << numer << ", denom = " << denom <<
        //     ", step = " << step.norm() << std::endl;

        // Update iterate if rho > 0
        if (rho > 0.0)
            gamma.noalias() += step;
        // std::cout << gamma.head(10).transpose() << std::endl;

        // Update mu
        if (rho < rho_t1)
        {
            mu = mu_s1 * mu;
        } else if (rho > rho_t2)
        {
            mu = std::max(mu_s2 * mu, kappa);
        }

        // Get the new f, g, H if gamma is updated
        if (rho > 0.0)
        {
#ifdef TIMING
            TimePoint clock_s5 = Clock::now();
#endif

            // If rho > 0, then newf < f, which means that T has been computed
            // in line selection
            f = prob.dual_obj_grad(gamma, g, T, false);

#ifdef TIMING
            TimePoint clock_s6 = Clock::now();
#endif

            gnorm = g.norm();
            delta = nu0 * std::pow(gnorm, pow_delta);
            prob.dual_sparsified_hess(T, g, delta, H.density(), H);

#ifdef TIMING
            TimePoint clock_s7 = Clock::now();
            cout << "[summary]=================================================" << std::endl;
            cout << "solve = " << (clock_s2 - clock_s1).count() <<
                ", grad = " << (clock_s6 - clock_s5).count() <<
                ", hess = " << (clock_s7 - clock_s6).count() << std::endl;
            cout << "line = " << (clock_s3 - clock_s2).count() <<
                ", rho = " << (clock_s4 - clock_s3).count() << std::endl;
            cout << "==========================================================" << std::endl;
#endif
        }

        if (verbose)
            cout << std::endl;

        // Record timing
        clock_t2 = Clock::now();

        // Collect progress statistics
        prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm);
        double duration = (clock_t2 - clock_t1).count();
        run_times.push_back(run_times.back() + duration);
        density.push_back(H.density());
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    result.prim_vals.swap(prim_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
    result.density.swap(density);
}

}  // namespace Sinkhorn
