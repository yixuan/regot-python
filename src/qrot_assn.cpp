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

void qrot_assn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double nu = 0.99, eta1 = 0.25, eta2 = 0.75,
        gamma1 = 2.0, gamma2 = 4.0, lam_min = 0.01, beta = 0.5,
        cg_tol = 1e-6;
    double xi = 0.0, lam = 1.0;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m), u(n + m), gu(n + m),
        v(n + m), gv(n + m), direc(n + m);

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
    for (i = 0; i < max_iter; i++)
    {
        if (verbose)
        {
            cout << "i = " << i << ", obj = " << f <<
                ", gnorm = " << gnorm << std::endl;
        }

        // Start timing
        clock_t1 = Clock::now();

        // Convergence test
        if (gnorm < tol)
            break;

        // Initialize xi
        if (i == 0)
        {
            xi = gnorm;
        }

        // Compute search direction
        const double mu = lam * gnorm;
        hess_cg(H, -g, mu, direc, cg_tol, verbose, cout);
        u.noalias() = gamma + direc;
        prob.dual_grad(u, gu);
        const double gunorm = gu.norm();

        // Successful Newton step
        if (gunorm <= nu * xi)
        {
            gamma.swap(u);
            xi = gunorm;
        } else {
            // Compute rho
            const double numer = -gu.dot(direc);
            const double denom = direc.squaredNorm();
            const double rho = numer / denom;

            // Compute projection direction
            v.noalias() = gamma + (-numer) / (gunorm * gunorm) * gu;
            prob.dual_grad(v, gv);
            const double gvnorm = gv.norm();

            if (rho >= eta1 && gvnorm <= gnorm)
            {
                // Projection step
                gamma.swap(v);
            } else {
                // Gradient descent step
                gamma.noalias() -= beta * g;
            }

            // xi is unchanged, update lam
            if (rho >= eta2)
            {
                lam = 0.5 * (lam_min + lam);
            } else if (rho >= eta1) {
                lam *= gamma1;
            } else {
                lam *= gamma2;
            }
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
