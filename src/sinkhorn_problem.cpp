#include "sinkhorn_problem.h"
#include "approx_proj.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefVec = Eigen::Ref<Vector>;
using RefConstVec = Eigen::Ref<const Vector>;
using RefMat = Eigen::Ref<Matrix>;
using RefConstMat = Eigen::Ref<const Matrix>;

inline double log_sum_exp(const double* data, const int n)
{
    double c = *(std::max_element(data, data + n));
    double res = 0.0;
    for (int i = 0; i < n; i++, data++)
        res += std::exp(*data - c);
    return c + std::log(res);
}

inline void log_sum_exp_rowwise(const RefConstMat& data, RefVec res)
{
    const int n = data.rows();
    const int m = data.cols();
    res.resize(n);
    Vector c = data.rowwise().maxCoeff();
    res.array() = (data - c.replicate(1, m)).array().exp().rowwise().sum();
    res.array() = c.array() + res.array().log();
}

inline void log_sum_exp_colwise(const RefConstMat& data, RefVec res)
{
    const int n = data.rows();
    const int m = data.cols();
    res.resize(m);
    Vector c = data.colwise().maxCoeff();
    res.array() = (data - c.transpose().replicate(n, 1)).array().exp().colwise().sum();
    res.array() = c.array() + res.array().log();
}

// Compute the primal objective function
double Problem::primal_val(const Vector& gamma) const
{
    // Extract betat and set beta=(betat, 0)
    Vector beta(m_m);
    beta.head(m_m - 1).noalias() = gamma.tail(m_m - 1);
    beta[m_m - 1] = 0.0;

    // Get transport plan
    // Compute T = exp((alpha (+) beta - M) / reg)
    Matrix T = (gamma.head(m_n).replicate(1, m_m) +
        beta.transpose().replicate(m_n, 1) - m_M) / m_reg;
    T.array() = T.array().exp();

    // Approximate projection
    Matrix plan_feas = approx_proj(T, m_a, m_b);

    double entropy = (plan_feas.array() * (1.0 - plan_feas.array().log())).sum();
    double prim_val = plan_feas.cwiseProduct(m_M).sum() - m_reg * entropy;
    return prim_val;
}

// Compute the objective function
// f(alpha, beta) = reg * sum(T) - a' * alpha - b' * beta
// T = exp((alpha (+) beta - M) / reg)
double Problem::dual_obj(const Vector& gamma) const
{
    // Compute T = exp((alpha (+) beta - M) / reg)
    Matrix T(m_n, m_m);
    compute_T(gamma, T);

    // Compute objective function value
    double Tsum = T.sum();
    double obj = m_reg * Tsum - gamma.head(m_n).dot(m_a) -
        gamma.tail(m_m - 1).dot(m_b.head(m_m - 1));

    return obj;
}

// Compute the gradient
void Problem::dual_grad(const Vector& gamma, Vector& grad) const
{
    // Compute T = exp((alpha (+) beta - M) / reg)
    Matrix T(m_n, m_m);
    compute_T(gamma, T);

    grad.resize(m_n + m_m - 1);
    // g(alpha) = T * 1m - a
    grad.head(m_n).noalias() = T.rowwise().sum() - m_a;
    // g(beta) = T' * 1n - b
    grad.tail(m_m - 1).noalias() = T.leftCols(m_m - 1).colwise().sum().transpose() - m_b;
}

// Compute the objective function and gradient
double Problem::dual_obj_grad(const Vector& gamma, Vector& grad) const
{
    // Call the second version below
    Matrix T(m_n, m_m);
    return dual_obj_grad(gamma, grad, T);
}

// Compute the objective function, gradient, and T
double Problem::dual_obj_grad(const Vector& gamma, Vector& grad, Matrix& T) const
{
    // Compute T = exp((alpha (+) beta - M) / reg)
    compute_T(gamma, T);

    grad.resize(m_n + m_m - 1);
    // g(alpha) = T * 1m - a
    grad.head(m_n).noalias() = T.rowwise().sum();
    double Tsum = grad.head(m_n).sum();
    grad.head(m_n).array() -= m_a.array();
    // g(beta) = T' * 1n - b
    grad.tail(m_m - 1).noalias() = T.leftCols(m_m - 1).colwise().sum().transpose() - m_b;

    // Compute objective function value
    double obj = m_reg * Tsum - gamma.head(m_n).dot(m_a) -
        gamma.tail(m_m - 1).dot(m_b.head(m_m - 1));

    return obj;
}

// Compute the objective function, gradient, and the true Hessian
void Problem::dual_obj_grad_densehess(
    const Vector& gamma, double& obj, Vector& grad, Matrix& hess
) const
{
    // Compute obj, grad, and T = exp((alpha (+) beta - M) / reg)
    Matrix T(m_n, m_m);
    obj = dual_obj_grad(gamma, grad, T);

    hess.resize(m_n + m_m - 1, m_n + m_m - 1);
    hess.setZero();

    // r = T * 1m, c = T' * 1n
    hess.diagonal().head(m_n).noalias() = T.rowwise().sum();
    hess.diagonal().tail(m_m - 1).noalias() = T.leftCols(m_m - 1).colwise().sum().transpose();

    // Off-diagonal elements
    hess.topRightCorner(m_n, m_m - 1).noalias() = T.leftCols(m_m - 1);
    hess.bottomLeftCorner(m_m - 1, m_n).noalias() = T.leftCols(m_m - 1).transpose();

    hess.array() /= m_reg;
}

// Compute the objective function, gradient, and sparsified Hessian
void Problem::dual_obj_grad_hess(
    const Vector& gamma, double delta, double& obj, Vector& grad, Hessian& hess
) const
{
    // Compute obj, grad, and T = exp((alpha (+) beta - M) / reg)
    Matrix T(m_n, m_m);
    obj = dual_obj_grad(gamma, grad, T);
    // Compute sparsified Hessian from T
    hess.compute_hess(T, m_reg, delta);
}

// Select a step size
double Problem::line_selection(
    const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
    double curobj, double& objval, bool verbose, std::ostream& cout
) const
{
    const int nc = static_cast<int>(candid.size());
    double best_step = 1.0;
    objval = std::numeric_limits<double>::infinity();
    for (int i = 0; i < nc; i++)
    {
        const double alpha = candid[i];
        Vector newgamma = gamma + alpha * direc;
        const double objfn = dual_obj(newgamma);
        if (objfn < objval)
        {
            best_step = alpha;
            objval = objfn;
        }
        if (objval < curobj)
        {
            return best_step;
        }
    }
    return best_step;
}

// Optimal beta given alpha
void Problem::optimal_beta(const RefConstVec& alpha, RefVec beta) const
{
    beta.resize(m_m);
    Matrix D(m_n, m_m);
    D.array() = (alpha.replicate(1, m_m) - m_M).array() / m_reg;
    log_sum_exp_colwise(D, beta);
    beta.noalias() = m_reg * (m_logb - beta);
}

// Optimal alpha given beta
void Problem::optimal_alpha(const RefConstVec& beta, RefVec alpha) const
{
    alpha.resize(m_n);
    Matrix D(m_n, m_m);
    D.array() = (beta.transpose().replicate(m_n, 1) - m_M).array() / m_reg;
    log_sum_exp_rowwise(D, alpha);
    alpha.noalias() = m_reg * (m_loga - alpha);
}

}  // namespace Sinkhorn
