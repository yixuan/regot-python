#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <LBFGS.h>
#include "timer.h"
#include "sinkhorn_problem.h"
#include "sinkhorn_result.h"
#include "sinkhorn_solvers.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

using LBFGSpp::LBFGSParam;
using LBFGSpp::LBFGSSolver;

class SinkhornDual
{
private:
    const Problem&  m_prob;
    int             m_iter;
    double          m_last_obj_val;
    Timer           m_timer;
    SinkhornResult& m_result;
    int             m_verbose;
    std::ostream&   m_cout;

public:
    SinkhornDual(const Problem& prob, SinkhornResult& result, std::ostream& cout):
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

    double operator()(const Vector& gamma, Vector& grad)
    {
        m_last_obj_val = m_prob.dual_obj_grad(gamma, grad);
        return m_last_obj_val;
    }

    void iterate(const Vector& gamma, const LBFGSSolver<double>& solver)
    {
        double duration = m_timer.toc("iter");

        // double prim_val = m_prob.primal_val(gamma);
        m_result.obj_vals.push_back(m_last_obj_val);
        // m_result.prim_vals.push_back(prim_val);
        m_result.mar_errs.push_back(solver.final_grad_norm());
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

void sinkhorn_lbfgs_dual_internal(
    SinkhornResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const SinkhornSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
)
{
    // Dimensions
    const int n = M.rows();
    const int m = M.cols();

    // Set up the problem
    Problem prob(M, a, b, reg);
    SinkhornDual dual(prob, result, cout);

    // L-BFGS parameters
    LBFGSParam<double> param;
    param.epsilon = tol;
    param.epsilon_rel = 0.0;
    param.max_iterations = max_iter;

    // Create solver
    LBFGSSolver<double> solver(param);

    // Initial guess
    double obj;
    Vector gamma(n + m - 1);
    if (opts.x0.size() == gamma.size())
    {
        gamma.noalias() = opts.x0;
    } else {
        gamma.head(n).setZero();
        Vector beta(m);
        prob.optimal_beta(gamma.head(n), beta);
        gamma.head(n).array() += beta[m - 1];
        gamma.tail(m - 1).array() = beta.head(m - 1).array() - beta[m - 1];
    }
    dual.reset();
    dual.set_verbose(verbose);
    int niter = solver.minimize(dual, gamma, obj);

    // Save results
    result.niter = niter;
    result.get_plan(gamma, prob);
    result.dual.swap(gamma);
}

}  // namespace Sinkhorn
