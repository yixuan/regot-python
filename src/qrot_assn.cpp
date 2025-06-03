#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "timer.h"
#include "qrot_hess.h"
#include "qrot_linsolve.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

void qrot_assn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Solver options
    double tau = opts.tau;
    int method = opts.method;

    // Algorithmic parameters
    constexpr double nu = 0.99, eta1 = 0.25, eta2 = 0.75,
        gamma1 = 2.0, gamma2 = 4.0, lam_min = 0.01, beta = 0.5,
        cg_tol = 1e-8;
    double xi = 0.0, lam = 1.0;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg, tau);
    Vector gamma(n + m), u(n + m), gu(n + m),
        v(n + m), gv(n + m), direc(n + m);

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
    Timer timer;
    timer.tic();
    // Initial objective function value, gradient, and Hessian
    double f;
    Vector g;
    Hessian H;
    prob.dual_obj_grad_hess(gamma, f, g, H);
    double gnorm = g.norm();
    // Record timing
    double duration = timer.toc("iter");

    // Collect progress statistics
    double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm / reg);
    run_times.push_back(duration);

    int i;
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

        // Initialize xi
        if (i == 0)
        {
            xi = gnorm;
        }

        // Compute search direction
        const double mu = lam * gnorm;
        lin_sol.solve(direc, H, -g, mu, true, cout);
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
        duration = timer.toc("iter");

        // Collect progress statistics
        prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm / reg);
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
