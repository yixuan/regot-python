#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <LBFGS.h>
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

using LBFGSpp::LBFGSParam;
using LBFGSpp::LBFGSSolver;

class QROTDual
{
private:
    const Problem& m_prob;
    int            m_iter;
    double         m_last_obj_val;
    TimePoint      m_clock_t1;
    TimePoint      m_clock_t2;
    QROTResult&    m_result;
    int            m_verbose;
    std::ostream&  m_cout;

public:
    QROTDual(const Problem& prob, QROTResult& result, std::ostream& cout):
        m_prob(prob), m_iter(0),
        m_last_obj_val(std::numeric_limits<double>::infinity()),
        m_result(result), m_verbose(0), m_cout(cout)
    {}

    void reset()
    {
        m_iter = 0;
        m_result.obj_vals.clear();
        m_result.obj_vals.reserve(1000);
        m_result.prim_vals.clear();
        m_result.prim_vals.reserve(1000);
        m_result.mar_errs.clear();
        m_result.mar_errs.reserve(1000);
        m_result.run_times.clear();
        m_result.run_times.reserve(1000);

        m_clock_t1 = Clock::now();
    }

    void set_verbose(int verbose) { m_verbose = verbose; }

    double operator()(const Vector& gamma, Vector& grad)
    {
        m_last_obj_val = m_prob.dual_obj_grad(gamma, grad);
        return m_last_obj_val;
    }

    void iterate(const Vector& gamma, const LBFGSSolver<double>& solver)
    {
        m_clock_t2 = Clock::now();

        double prim_val = m_prob.primal_val(gamma);
        m_result.obj_vals.push_back(m_last_obj_val);
        m_result.prim_vals.push_back(prim_val);
        m_result.mar_errs.push_back(solver.final_grad_norm() / m_prob.reg());
        double current = m_result.run_times.empty() ? 0.0 : m_result.run_times.back();
        double duration = (m_clock_t2 - m_clock_t1).count();
        m_result.run_times.push_back(current + duration);

        if (m_verbose >= 1)
        {
            m_cout << "iter = " << m_iter << ", objval = " << m_last_obj_val <<
                ", ||grad|| = " << solver.final_grad_norm() << std::endl;
        }

        m_iter++;
        m_clock_t1 = Clock::now();
    }
};

void qrot_lbfgs_dual_internal(
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

    // Set up the problem
    Problem prob(M, a, b, reg, tau);
    QROTDual dual(prob, result, cout);

    // L-BFGS parameters
    LBFGSParam<double> param;
    param.epsilon = tol;
    param.epsilon_rel = tol;
    param.max_iterations = max_iter;

    // Create solver
    LBFGSSolver<double> solver(param);

    // Initial guess
    double obj;
    Vector gamma(n + m);
    if (opts.x0.size() == gamma.size())
    {
        gamma.noalias() = opts.x0;
    } else {
        gamma.head(n).setZero();
        prob.optimal_beta(gamma.head(n), gamma.tail(m));
    }
    dual.reset();
    dual.set_verbose(verbose);
    int niter = solver.minimize(dual, gamma, obj);

    // Save results
    result.niter = niter;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
}

}  // namespace QROT
