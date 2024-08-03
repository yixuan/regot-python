#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

// Whether to print detailed timing information
// #define TIMING 1

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

void sinkhorn_bcd_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg);
    Vector alpha(n), beta(m), gamma(n + m - 1), grad(n + m - 1);

    // Progress statistics
    std::vector<double> obj_vals;
    std::vector<double> prim_vals;
    std::vector<double> mar_errs;
    std::vector<double> run_times;

    // Initial value
    if (opts.x0.size() == gamma.size())
    {
        alpha.noalias() = opts.x0.head(n);
        beta.head(m - 1).noalias() = opts.x0.tail(m - 1);
        beta[m - 1] = 0.0;
    } else {
        alpha.setZero();
        prob.optimal_beta(alpha, beta);
    }

    // Start timing
    TimePoint clock_t1 = Clock::now();
    // Initial objective function and gradient
    gamma.head(n).array() = alpha.array() + beta[m - 1];
    gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];
    double f = prob.dual_obj_grad(gamma, grad);
    double gnorm = grad.norm();
    // Record timing
    TimePoint clock_t2 = Clock::now();

    // Collect progress statistics
    double prim_val = prob.primal_val(gamma);
    obj_vals.push_back(f);
    prim_vals.push_back(prim_val);
    mar_errs.push_back(gnorm);
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

        // Optimal alpha given beta
        prob.optimal_alpha(beta, alpha);

#ifdef TIMING
        TimePoint clock_s1 = Clock::now();
#endif

        // Optimal beta given alpha
        prob.optimal_beta(alpha, beta);

#ifdef TIMING
        TimePoint clock_s2 = Clock::now();
#endif

        gamma.head(n).array() = alpha.array() + beta[m - 1];
        gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];

        // Get the new f and g
        f = prob.dual_obj_grad(gamma, grad);
        gnorm = grad.norm();
        // Record timing
        clock_t2 = Clock::now();

#ifdef TIMING
        cout << "[summary]=================================================" << std::endl;
        cout << "alpha = " << (clock_s1 - clock_t1).count() <<
            ", beta = " << (clock_s2 - clock_s1).count() <<
            ", grad = " << (clock_t2 - clock_s2).count() << std::endl;
        cout << "==========================================================" << std::endl << std::endl;
#endif

        // Collect progress statistics
        prim_val = prob.primal_val(gamma);
        obj_vals.push_back(f);
        prim_vals.push_back(prim_val);
        mar_errs.push_back(gnorm);
        double duration = (clock_t2 - clock_t1).count();
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

}  // namespace Sinkhorn
