#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "LBFGS.h"
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

class QROTSemiDual
{
private:
    const Problem& m_prob;
    double         m_last_obj_val;
    TimePoint      m_clock_start;
    QROTResult&    m_result;

public:
    QROTSemiDual(const Problem& prob, QROTResult& result):
        m_prob(prob),
        m_last_obj_val(std::numeric_limits<double>::infinity()),
        m_result(result)
    {}

    void reset()
    {
        m_result.obj_vals.clear();
        m_result.obj_vals.reserve(1000);
        m_result.mar_errs.clear();
        m_result.mar_errs.reserve(1000);
        m_result.run_times.clear();
        m_result.run_times.reserve(1000);

        m_clock_start = Clock::now();
    }

    double operator()(const Vector& alpha, Vector& grad)
    {
        m_last_obj_val = m_prob.semi_dual_obj_grad(alpha, grad);
        return m_last_obj_val;
    }

    void iterate(const LBFGSSolver<double>& solver)
    {
        m_result.obj_vals.push_back(m_last_obj_val);
        m_result.mar_errs.push_back(solver.final_grad_norm() / m_prob.reg());
        TimePoint now = Clock::now();
        m_result.run_times.push_back((now - m_clock_start).count());
    }
};

void qrot_lbfgs_semi_dual_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, bool verbose, std::ostream& cout
)
{
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg);
    QROTSemiDual semi_dual(prob, result);

    // L-BFGS parameters
    LBFGSParam<double> param;
    param.epsilon = tol;
    param.epsilon_rel = tol;
    param.max_iterations = max_iter;

    // Create solver
    LBFGSSolver<double> solver(param);

    // Initial guess
    double obj;
    Vector alpha = Vector::Zero(n);
    semi_dual.reset();
    int niter = solver.minimize(semi_dual, alpha, obj);

    Vector gamma(n + m);
    gamma.head(n).noalias() = alpha;
    prob.optimal_beta(gamma.head(n), gamma.tail(m));

    // Save results
    result.niter = niter;
    result.get_plan(gamma, prob);
}
