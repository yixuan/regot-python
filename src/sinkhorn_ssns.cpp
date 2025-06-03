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

void sinkhorn_ssns_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
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
    Timer timer, timer_inner;
    timer.tic();
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
    double duration = timer.toc("iter");

    // Collect progress statistics
    // double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    // prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm);
    run_times.push_back(duration);
    density.push_back(H.density());

    int i;
    std::vector<double> alphas{1.0, 0.5, 0.25, 0.1};
    // std::vector<double> alphas_small{0.05, 0.01, 0.001, 1e-4};
    // bool use_alpha_small = true;
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

        // Compute search direction
        double shift = mu * std::pow(gnorm, pow_lam);
        if (verbose >= 2)
        {
            cout << "[params]===================================================" << std::endl;
            cout << "║ rho = " << rho << ", mu = " << mu << ", shift = " << shift << std::endl;
            cout << "║----------------------------------------------------------" << std::endl;
        }
        if (rho <= 0.0)
        {
            // No update in previous iteration
            // Use last direction for warm start
            lin_sol.cg_x0.noalias() = direc.tail(m);
        } else {
            lin_sol.cg_x0.setZero();
        }
        bool same_sparsity = (rho <= 0.0);
        timer_inner.tic();
        lin_sol.solve(direc, H, -g, shift, !same_sparsity, cout);
        timer_inner.toc("lin_solve");

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
        timer_inner.toc("line_search");

        // Estimate rho
        step.noalias() = alpha * direc;
        H.apply_Hsx(step, Hstep);
        timer_inner.toc("rho");

        const double numer = f - newf;
        const double denom = -g.dot(step) - 0.5 * step.dot(Hstep);
        rho = numer / denom;
        if (verbose >= 2)
        {
            cout << "[update]---------------------------------------------------" << std::endl;
            cout << "║ alpha = " << alpha << ", ||step|| = " << step.norm() << std::endl;
            cout << "║ rho_numer = " << numer << ", rho_denom = " << denom << std::endl;
        }

        // Update iterate if rho > 0
        if (rho > 0.0)
            gamma.noalias() += step;
        // cout << gamma.head(10).transpose() << std::endl;

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
            timer_inner.tic();
            // If rho > 0, then newf < f, which means that T has been computed
            // in line selection
            f = prob.dual_obj_grad(gamma, g, T, false);
            timer_inner.toc("grad");

            gnorm = g.norm();
            delta = nu0 * std::pow(gnorm, pow_delta);
            double density_hint = H.density();
            prob.dual_sparsified_hess(T, g, delta, H.density(), H);
            timer_inner.toc("hess");

            if (verbose >= 2)
            {
                cout << "║----------------------------------------------------------" << std::endl;
                cout << "[sparse]---------------------------------------------------" << std::endl;
                cout << "║ delta = " << delta << ", den = " << density_hint << ", den_new = " << H.density() << std::endl;
                cout << "║----------------------------------------------------------" << std::endl;
                cout << "[timing]---------------------------------------------------" << std::endl;
                cout << "║ lin_solve = " << timer_inner["lin_solve"] <<
                    ", grad = " << timer_inner["grad"] <<
                    ", hess = " << timer_inner["hess"] << std::endl;
                cout << "║ line_search = " << timer_inner["line_search"] <<
                    ", rho = " << timer_inner["rho"] << std::endl;
             }
        }
        if (verbose >= 2)
            cout << "===========================================================" << std::endl << std::endl;

        // Record timing
        duration = timer.toc("iter");

        // Collect progress statistics
        // prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        // prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm);
        run_times.push_back(run_times.back() + duration);
        density.push_back(H.density());
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    // result.prim_vals.swap(prim_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
    result.density.swap(density);
}

}  // namespace Sinkhorn
