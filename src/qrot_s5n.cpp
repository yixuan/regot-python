#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "qrot_hess.h"
#include "qrot_linsolve.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void qrot_s5n_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Solver options
    double tau = opts.tau;
    int method = opts.method;
    double mu0 = opts.mu0;

    // Algorithmic parameters
    constexpr double rho_t1 = 0.25, rho_t2 = 0.75,
        mu_s1 = 4.0, mu_s2 = 0.5,
        delta1 = 1.0, delta2 = 1.0,
        cg_tol = 1e-8;

    double delta = delta1, kappa = 0.001, mu = mu0, rho = 1.0;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg, tau);
    Vector gamma(n + m), step(n + m), Hstep(n + m), direc(n + m);
    Vector cg_x0(m);

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

    // Linear solver
    QROTLinearSolver lin_sol;
    lin_sol.method = method;
    lin_sol.tau = tau;
    lin_sol.cg_tol = cg_tol;
    lin_sol.verbose = verbose;

    // Start timing
    TimePoint clock_t1 = Clock::now();
    // Initial objective function value, gradient, and Hessian
    double f;
    Vector g;
    Hessian H;
    prob.dual_obj_grad_hess(gamma, f, g, H);
    double gnorm = g.norm();
    // Record timing
    TimePoint clock_t2 = Clock::now();

    // Collect progress statistics
    double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm / reg);
    run_times.push_back((clock_t2 - clock_t1).count());

    int i;
    std::vector<double> alphas{1.0, 0.5, 0.1};
    std::vector<double> alphas_small{0.05, 0.01, 0.001, 1e-4};
    bool use_alpha_small = true;
    for (i = 0; i < max_iter; i++)
    {
        if (verbose)
        {
            cout << "i = " << i << ", obj = " << f <<
                ", gnorm = " << gnorm << ", rho = " << rho <<
                ", mu = " << mu << std::endl;
        }
        double dgap = prim_val + f / reg;
        // cout << "log_dual_gap = " << std::log10(dgap);

        // Start timing
        clock_t1 = Clock::now();

        // Convergence test
        if (gnorm < tol)
            break;

        // Compute search direction
        // double shift = mu * std::pow(gnorm, delta);
        double shift = mu * std::pow(dgap, delta);
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
        lin_sol.solve(direc, H, -g, shift, !same_sparsity, cout);

        // Step size selection
        double newf;
        double alpha = prob.line_selection2(
            alphas, gamma, direc, f, newf, verbose, cout);
        if (newf >= f)
        {
            if (use_alpha_small)
            {
                alpha = prob.line_selection2(
                    alphas_small, gamma, direc, f, newf, verbose, cout);
            }
            use_alpha_small = !use_alpha_small;
        }
        else
        {
            use_alpha_small = true;
        }

        // Estimate rho
        step.noalias() = alpha * direc;
        H.apply_Hx(step, 0.0, tau, Hstep);
        const double numer = f - newf;
        const double denom = -g.dot(step) - 0.5 * step.dot(Hstep);
        rho = numer / denom;

        // Update iterate if rho > 0
        if (rho > 0.0)
            gamma.noalias() += step;

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
            prob.dual_obj_grad_hess(gamma, f, g, H);
            gnorm = g.norm();
        }

        // Record timing
        clock_t2 = Clock::now();

        // Collect progress statistics
        prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm / reg);
        double duration = (clock_t2 - clock_t1).count();
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
