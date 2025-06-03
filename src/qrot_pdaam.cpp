#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "timer.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

void qrot_pdaam_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double L0 = 1.0;
    constexpr int max_inner = 20;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg, 0.0);
    Vector gamma(n + m), grad(n + m);
    // gamma = (alpha, beta)
    // lambda = -alpha, mu = -beta, dual = (lambda, mu) = -gamma
    // phi = -obj / reg, d_phi/d_dual = grad / reg
    Vector dual(n + m), dual_new(n + m), dualp(n + m), dualt(n + m);
    QROTResult::ResultMatrix plan(n, m);

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
    dual.noalias() = -gamma;
    dualp.noalias() = dual;
    dualt.noalias() = dual;
    result.get_plan(gamma, prob);

    // Start timing
    Timer timer;
    timer.tic();
    // Initial objective function and gradient
    double f = prob.dual_obj_grad(gamma, grad);
    double gnorm = grad.norm();
    // Record timing
    double duration = timer.toc("iter");

    // Collect progress statistics
    double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm / reg);
    run_times.push_back(duration);

    int i;
    double L = L0, palpha = 0.0;
    for (i = 0; i < max_iter; i++)
    {
        // Start timing
        timer.tic();

        // Line search
        if (verbose >= 2)
        {
            cout << "\n[line search]=================================================" << std::endl;
        }
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
            f = prob.dual_obj_grad(gamma, grad);
            const double phit = -f / reg;

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
            f = prob.dual_obj(gamma);
            phi_new = -f / reg;

            // Line search test
            const double rhs = phit + (lam_resid + mu_resid) / (2.0 * L_new * reg * reg);
            if (verbose >= 2)
            {
                cout << "║ j = " << j << ", L = " << L_new <<
                    ", palpha = " << palpha << std::endl;
                cout << "║ phit = " << phit << ", phi_new = " << phi_new <<
                    ", rhs = " << rhs << std::endl;
                cout << "║-------------------------------------------------------------" << std::endl;
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

        // Record timing
        duration = timer.toc("iter");

        // Collect progress statistics
        prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        prim_vals.push_back(prim_val);
        mar_errs.push_back(mar_err);
        run_times.push_back(run_times.back() + duration);

        if (verbose >= 1)
        {
            cout << "iter = " << i << ", primal_obj = " << primal_obj <<
                ", phi_new = " << phi_new <<
                    ", mar_err = " << mar_err << std::endl;
        }

        // Convergence test
        if (primal_obj - phi_new <= tol && mar_err <= tol)
            break;
    }

    // Save results
    result.niter = i;
    result.dual.swap(gamma);
    result.obj_vals.swap(obj_vals);
    result.prim_vals.swap(prim_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}

}  // namespace QROT
