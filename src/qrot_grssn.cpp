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

void qrot_grssn_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, double shift,
    bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double theta = 0.5, kappa = 0.5, cg_tol = 1e-6;
    constexpr int nlinesearch = 20;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m), newgamma(n + m), direc(n + m);

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

        // Compute search direction
        hess_cg(H, -g, shift, direc, cg_tol, verbose, cout);

        // Line search
        double step = 1.0;
        for (int k = 0; k < nlinesearch; k++)
        {
            newgamma.noalias() = gamma + step * direc;
            const double newf = prob.dual_obj(newgamma);
            if (newf <= f + theta * step * g.dot(direc))
                break;
            step *= kappa;
        }
        gamma.swap(newgamma);

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
