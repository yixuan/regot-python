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

void qrot_bcd_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg);
    Vector gamma(n + m), grad(n + m);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    gamma.head(n).setZero();
    prob.optimal_beta(gamma.head(n), gamma.tail(m));

    int i;
    // Start timing
    TimePoint clock_start = Clock::now();
    for (i = 0; i < max_iter; i++)
    {
        // Get the current f, g
        double obj = prob.dual_obj_grad(gamma, grad);
        double gnorm = grad.norm();
        // Record timing after each iteration
        TimePoint now = Clock::now();

        // Collect progress statistics
        obj_vals.push_back(obj);
        mar_errs.push_back(gnorm / reg);
        run_times.push_back((now - clock_start).count());
        if (verbose)
        {
            cout << "i = " << i << ", obj = " << obj <<
                ", gnorm = " << gnorm << std::endl;
        }
        // Convergence test
        if (gnorm < tol)
            break;

        // Optimal alpha given beta
        prob.optimal_alpha(gamma.tail(m), gamma.head(n));
        // Optimal beta given alpha
        prob.optimal_beta(gamma.head(n), gamma.tail(m));
    }

    // Save results
    result.niter = i;
    result.get_plan(gamma, prob);
    result.obj_vals.swap(obj_vals);
    result.mar_errs.swap(mar_errs);
    result.run_times.swap(run_times);
}
