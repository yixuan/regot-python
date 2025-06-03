#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <LBFGS.h>
#include "timer.h"
#include "qrot_problem.h"
#include "qrot_result.h"
#include "qrot_solvers.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

using LBFGSpp::LBFGSParam;
using LBFGSpp::LBFGSSolver;

class QROTSemiDual
{
private:
    const Problem& m_prob;
    int            m_iter;
    double         m_last_obj_val;
    Timer          m_timer;
    QROTResult&    m_result;
    int            m_verbose;
    std::ostream&  m_cout;

public:
    QROTSemiDual(const Problem& prob, QROTResult& result, std::ostream& cout):
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

        m_timer.tic();
    }

    void set_verbose(int verbose) { m_verbose = verbose; }

    double operator()(const Vector& alpha, Vector& grad)
    {
        m_last_obj_val = m_prob.semi_dual_obj_grad(alpha, grad);
        return m_last_obj_val;
    }

    void iterate(const Vector& alpha, const LBFGSSolver<double>& solver)
    {
        double duration = m_timer.toc("iter");

        const int n = m_prob.size_n();
        const int m = m_prob.size_m();
        Vector gamma(n + m);
        gamma.head(n).noalias() = alpha;
        m_prob.optimal_beta(gamma.head(n), gamma.tail(m));
        double prim_val = m_prob.primal_val(gamma);
        m_result.obj_vals.push_back(m_last_obj_val);
        m_result.prim_vals.push_back(prim_val);
        m_result.mar_errs.push_back(solver.final_grad_norm() / m_prob.reg());
        double current = m_result.run_times.empty() ? 0.0 : m_result.run_times.back();
        m_result.run_times.push_back(current + duration);

        if (m_verbose >= 1)
        {
            m_cout << "iter = " << m_iter << ", objval = " << m_last_obj_val <<
                ", ||grad|| = " << solver.final_grad_norm() << std::endl;
        }

        m_iter++;
        m_timer.tic();
    }
};

void qrot_lbfgs_semi_dual_internal(
    QROTResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const QROTSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg, 0.0);
    QROTSemiDual semi_dual(prob, result, cout);

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
    if (opts.x0.size() == n + m)
    {
        alpha.noalias() = opts.x0.head(n);
    }
    semi_dual.reset();
    semi_dual.set_verbose(verbose);
    int niter = solver.minimize(semi_dual, alpha, obj);

    Vector gamma(n + m);
    gamma.head(n).noalias() = alpha;
    prob.optimal_beta(gamma.head(n), gamma.tail(m));

    // Save results
    result.niter = niter;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
}

}  // namespace QROT
