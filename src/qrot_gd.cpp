#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "qrot_hess.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void qrot_gd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Algorithmic parameters
    constexpr double theta = 0.5, kappa = 0.5;
    constexpr int nlinesearch = 20;

    // Dual variables and intermediate variables
    Problem prob(M, a, b, reg);
    Vector gamma(n + m), newgamma(n + m);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    gamma.head(n).setZero();
    prob.optimal_beta(gamma.head(n), gamma.tail(m));

    // Objective function value, gradient, and Hessian
    double f;
    Vector g;

    int i;
    // Start timing
    TimePoint clock_start = Clock::now();
    for (i = 0; i < max_iter; i++)
    {
        // Get the current f, g
        f = prob.dual_obj_grad(gamma, g);
        // Record timing after each iteration
        TimePoint now = Clock::now();

        // Collect progress statistics
        double gnorm = g.norm();
        obj_vals.push_back(f);
        mar_errs.push_back(gnorm / reg);
        run_times.push_back((now - clock_start).count());
        if (verbose)
        {
            cout << "i = " << i << ", obj = " << f <<
                ", gnorm = " << gnorm << std::endl;
        }
        // Convergence test
        if (gnorm < tol)
            break;

        // Line search
        double step = 1.0;
        for (int k = 0; k < nlinesearch; k++)
        {
            newgamma.noalias() = gamma - step * g;
            const double newf = prob.dual_obj(newgamma);
            if (newf <= f - theta * step * g.squaredNorm())
                break;
            step *= kappa;
        }
        gamma.swap(newgamma);
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.obj_vals.swap(obj_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}
