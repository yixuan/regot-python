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

void qrot_grssn_internal(
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
    double shift = opts.shift;
    int method = opts.method;

    // Algorithmic parameters
    constexpr double theta = 0.5, kappa = 0.5, cg_tol = 1e-8;
    constexpr int nlinesearch = 20;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg, tau);
    Vector gamma(n + m), newgamma(n + m), direc(n + m);

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

        // Compute search direction
        lin_sol.solve(direc, H, -g, shift, true, cout);

        // Line search
        double step = 1.0;
        double thresh = theta * g.dot(direc);
        for (int k = 0; k < nlinesearch; k++)
        {
            newgamma.noalias() = gamma + step * direc;
            const double newf = prob.dual_obj(newgamma);
            if (newf <= f + step * thresh)
                break;
            step *= kappa;
        }
        gamma.swap(newgamma);

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
