#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "qrot_hess.h"
#include "qrot_cg.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void qrot_s5n_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double rho_t1 = 0.25, rho_t2 = 0.75,
        mu_s1 = 4.0, mu_s2 = 0.5,
        delta1 = 1.5, delta2 = 1.0,
        cg_tol = 1e-6;

    double delta = delta1, kappa = 0.001, mu = 1.0, rho = 0.0;
    bool stage2 = false;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m), step(n + m), Hstep(n + m), direc(n + m);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    gamma.head(n).setZero();
    prob.optimal_beta(gamma.head(n), gamma.tail(m));

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
    std::vector<double> alphas{1.0, 0.5, 0.25, 0.125, 0.0625, 0.01};
    for (i = 0; i < max_iter; i++)
    {
        if (verbose)
        {
            cout << "i = " << i << ", obj = " << f <<
                ", gnorm = " << gnorm << ", rho = " << rho <<
                ", mu = " << mu << std::endl;
        }

        // Start timing
        clock_t1 = Clock::now();

        // Convergence test
        if (gnorm < tol)
            break;

        // Compute search direction
        double shift = mu * std::pow(gnorm, delta);
        // Initial conservative steps
        if (i < 10)
        {
            shift = 1e-3 * std::max(H.h1().maxCoeff(), H.h2().maxCoeff());
            mu = 1.0;
        }
        // When shift becomes small for the first time,
        // change kappa and delta to safer values
        if ((!stage2) && shift < 1e-6 * std::max(H.h1().maxCoeff(), H.h2().maxCoeff()))
        {
            if (verbose)
                cout << "\n*********** Stage 2 ***********\n" << std::endl;
            stage2 = true;
            delta = delta2;
            kappa = shift / std::pow(gnorm, delta);
            shift = kappa * std::pow(gnorm, delta);
        }
        hess_cg(H, -g, shift, direc, cg_tol, verbose, cout);

        // Step size selection
        double newf;
        double alpha = prob.line_selection2(
            alphas, gamma, direc, f, newf, verbose, cout);

        // Estimate rho
        step.noalias() = alpha * direc;
        H.apply_Hx(step, 0.0, Hstep);
        const double numer = f - newf;
        const double denom = -g.dot(step) - 0.5 * step.dot(Hstep);
        rho = numer / denom;

        // In theoretical analysis, we let rho > 0, but here we can
        // allow for a small increase in objective function value
        if (rho > -1.0)
            gamma.noalias() += step;

        // Update mu
        if (rho < rho_t1)
        {
            mu = mu_s1 * mu;
        } else if (rho > rho_t2)
        {
            mu = std::max(mu_s2 * mu, kappa);
        }

        // Get the new f, g, H
        prob.dual_obj_grad_hess(gamma, f, g, H);
        gnorm = g.norm();
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
    result.obj_vals.swap(obj_vals);
    result.prim_vals.swap(prim_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}
