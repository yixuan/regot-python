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

void qrot_apdagd_internal(
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
    Vector dual(n + m), dual_new(n + m), dualp(n + m), dualp_new(n + m), dualt(n + m);
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
    double L = L0, pbeta = 0.0;
    for (i = 0; i < max_iter; i++)
    {
        // Line search
        L *= 0.5;
        double ptau = 0.0, phi_new = 0.0;
        for (int j = 0; j < max_inner; j++)
        {
            // Notations in different papers
            // eta - dual, zeta - dualp, lambda - dualt

            // Compute palpha and new pbeta
            // (step sizes, not dual variables)
            const double palpha = (1.0 + std::sqrt(4.0 * L * pbeta + 1.0)) / (2.0 * L);
            const double pbeta_new = pbeta + palpha;
            ptau = palpha / pbeta_new;

            // Compute the new dual~, phi(dual~), and d_phi(dual~)
            dualt.noalias() = (1.0 - ptau) * dual + ptau * dualp;
            gamma.noalias() = -dualt;
            obj = prob.dual_obj_grad(gamma, grad);
            const double phit = -obj / reg;

            // Compute the new dual'
            dualp_new.noalias() = dualp + (palpha / reg) * grad;

            // Compute the new dual
            dual_new.noalias() = (1.0 - ptau) * dual + ptau * dualp_new;
            gamma.noalias() = -dual_new;
            obj = prob.dual_obj(gamma);
            phi_new = -obj / reg;

            // Line search test
            const double rhs = phit + grad.dot(dual_new - dualt) / reg -
                0.5 * L * (dual_new - dualt).squaredNorm();
            if (verbose)
            {
                cout << "i = " << i << ", j = " << j << ", palpha = " << palpha <<
                    ", pbeta = " << pbeta_new << ", phit = " << phit <<
                    ", phi_new = " << phi_new <<
                    ", rhs = " << rhs << std::endl;
            }
            if (phi_new >= rhs || j >= max_inner - 1)
            {
                // Update variables and exit the line search
                pbeta = pbeta_new;
                dualp.swap(dualp_new);
                dual.swap(dual_new);
                break;
            }
            L *= 2.0;
        }

        // Save old plan
        plan.swap(result.plan);
        // Compute new plan
        gamma.noalias() = -dualt;
        result.get_plan(gamma, prob);
        // Update plan
        result.plan.noalias() = (1.0 - ptau) * plan + ptau * result.plan;

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
