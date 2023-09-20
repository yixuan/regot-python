#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void qrot_pdaam_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double L0 = 1.0;
    constexpr int max_inner = 20;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m), grad(n + m);
    // gamma = (alpha, beta)
    // lambda = -alpha, mu = -beta, dual = (lambda, mu) = -gamma
    // phi = -obj / reg, d_phi/d_dual = grad / reg
    Vector dual(n + m), dual_new(n + m), dualp(n + m), dualt(n + m);
    QROTResult::ResultMatrix plan(n, m);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    gamma.head(n).setZero();
    prob.optimal_beta(gamma.head(n), gamma.tail(m));
    dual.noalias() = -gamma;
    dualp.noalias() = dual;
    dualt.noalias() = dual;
    result.get_plan(gamma, prob);

    // Start timing
    TimePoint clock_start = Clock::now();
    // Get the current f, g
    double obj = prob.dual_obj_grad(gamma, grad);
    double gnorm = grad.norm();
    TimePoint now = Clock::now();
    // Collect progress statistics
    obj_vals.push_back(obj);
    mar_errs.push_back(gnorm / reg);
    run_times.push_back((now - clock_start).count());

    int i;
    double L = L0, palpha = 0.0;
    for (i = 0; i < max_iter; i++)
    {
        // Line search
        double L_new = 0.5 * L, phi_new = 0.0, tau_old_plan = 0.0, tau_new_plan = 0.0;
        for (int j = 0; j < max_inner; j++)
        {
            // Notations in different papers
            // eta - dual, zeta - dualp, lambda - dualt

            // Compute new palpha (step size, not dual variable)
            const double palpha_new = (1.0 + std::sqrt(4.0 * palpha * palpha * L * L_new + 1.0)) / (2.0 * L_new);
            const double ptau = 1.0 / (palpha_new * L_new);

            // Compute the new dual~ and compute phi(dual~)
            dualt.noalias() = (1.0 - ptau) * dual + ptau * dualp;
            gamma.noalias() = -dualt;
            obj = prob.dual_obj_grad(gamma, grad);
            const double phit = -obj / reg;

            // Compare residuals and choose which block to update
            const double lam_resid = grad.head(n).squaredNorm();
            const double mu_resid = grad.tail(m).squaredNorm();
            if (lam_resid > mu_resid)
            {
                // Optimal alpha given beta
                prob.optimal_alpha(gamma.tail(m), gamma.head(n));
            } else {
                // Optimal beta given alpha
                prob.optimal_beta(gamma.head(n), gamma.tail(m));
            }
            dual_new.noalias() = -gamma;
            obj = prob.dual_obj(gamma);
            phi_new = -obj / reg;

            // Line search test
            const double rhs = phit + (lam_resid + mu_resid) / (2.0 * L_new * reg * reg);
            if (verbose)
            {
                cout << "i = " << i << ", j = " << j <<
                    ", L = " << L_new <<
                    ", palpha = " << palpha_new <<
                    ", phit = " << phit << ", phi_new = " << phi_new <<
                    ", rhs = " << rhs << std::endl;
            }
            if (phi_new >= rhs || j >= max_inner - 1)
            {
                // Update variables and exit the line search
                const double palpha_ratio = palpha / palpha_new;
                const double L_ratio = L / L_new;
                tau_old_plan = palpha_ratio * palpha_ratio * L_ratio;
                tau_new_plan = ptau;
                palpha = palpha_new;
                L = L_new;
                dual.swap(dual_new);
                break;
            }
            L_new *= 2.0;
        }

        // Compute the new dual'
        dualp.noalias() += (palpha / reg) * grad;

        // Save old plan
        plan.swap(result.plan);
        // Compute new plan
        gamma.noalias() = -dualt;
        result.get_plan(gamma, prob);
        // Update plan
        result.plan.noalias() = tau_old_plan * plan + tau_new_plan * result.plan;

        // Statistics for convergence test
        const double primal_obj = (M.cwiseProduct(result.plan)).sum() +
            0.5 * reg * result.plan.squaredNorm();
        Vector resid_a = result.plan.rowwise().sum() - a;
        Vector resid_b = result.plan.colwise().sum().transpose() - b;
        const double mar_err = std::sqrt(resid_a.squaredNorm() + resid_b.squaredNorm());

        // Record timing after each iteration
        now = Clock::now();

        // Collect progress statistics
        obj_vals.push_back(obj);
        mar_errs.push_back(mar_err);
        run_times.push_back((now - clock_start).count());
        if (verbose)
        {
            cout << "i = " << i << ", primal_obj = " << primal_obj <<
                ", phi_new = " << phi_new <<
                ", mar_err = " << mar_err << std::endl << std::endl;
        }

        // Convergence test
        if (primal_obj - phi_new <= tol && mar_err <= tol)
            break;
    }

    // Save results
    result.niter = i;
    result.obj_vals.swap(obj_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}
