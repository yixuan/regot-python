#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <LBFGS.h>
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

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
    double         m_last_obj_val;
    TimePoint      m_clock_t1;
    TimePoint      m_clock_t2;
    QROTResult&    m_result;

public:
    QROTDual(const Problem& prob, QROTResult& result):
        m_prob(prob),
        m_last_obj_val(std::numeric_limits<double>::infinity()),
        m_result(result)
    {}

    void reset()
    {
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

        m_clock_t1 = Clock::now();
    }
};

void qrot_lbfgs_dual_internal(
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
    QROTDual dual(prob, result);

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
    gamma.head(n).setZero();
    prob.optimal_beta(gamma.head(n), gamma.tail(m));
    dual.reset();
    int niter = solver.minimize(dual, gamma, obj);

    // Save results
    result.niter = niter;
    result.get_plan(gamma, prob);
}
