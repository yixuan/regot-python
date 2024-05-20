#ifndef REGOT_SINKHORN_PROBLEM_H
#define REGOT_SINKHORN_PROBLEM_H

#include <Eigen/Core>
#include "sinkhorn_hess.h"

namespace Sinkhorn {

class Problem
{
private:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    using RefVec = Eigen::Ref<Vector>;
    using RefConstVec = Eigen::Ref<const Vector>;
    using RefMat = Eigen::Ref<Matrix>;
    using RefConstMat = Eigen::Ref<const Matrix>;

    const int    m_n;
    const int    m_m;
    RefConstMat  m_M;
    RefConstVec  m_a;
    RefConstVec  m_b;
    const double m_reg;
    Vector       m_loga;
    Vector       m_logb;

    // Compute T = exp((alpha (+) beta - M) / reg)
    inline void compute_T(const Vector& gamma, Matrix& T) const
    {
        T.resize(m_n, m_m);

        // Extract betat and set beta=(betat, 0)
        Vector beta(m_m);
        beta.head(m_m - 1).noalias() = gamma.tail(m_m - 1);
        beta[m_m - 1] = 0.0;

        // Compute T
        T.noalias() = (gamma.head(m_n).replicate(1, m_m) +
            beta.transpose().replicate(m_n, 1) - m_M) / m_reg;
        T.array() = T.array().exp();
    }

public:
    Problem(const RefConstMat& M, const RefConstVec& a, const RefConstVec& b, double reg):
        m_n(M.rows()), m_m(M.cols()),
        m_M(M), m_a(a), m_b(b),
        m_reg(reg), m_loga(m_n), m_logb(m_m)
    {
        // log(a) and log(b)
        m_loga.array() = m_a.array().log();
        m_logb.array() = m_b.array().log();
    }

    // Return the dimensions
    int size_n() const { return m_n; }
    int size_m() const { return m_m; }

    // Return the matrices/vectors
    RefConstMat get_M() const { return m_M; }
    RefConstVec get_a() const { return m_a; }
    RefConstVec get_b() const { return m_b; }

    // Return the regularization parameter
    double reg() const { return m_reg; }

    // Compute the primal objective function
    double primal_val(const Vector& gamma) const;

    // Compute the objective function
    // f(alpha, beta) = reg * sum(T) - a' * alpha - b' * beta
    // T = exp((alpha (+) beta - M) / reg)
    double dual_obj(const Vector& gamma) const;

    // Compute the gradient
    void dual_grad(const Vector& gamma, Vector& grad) const;

    // Compute the objective function and gradient
    double dual_obj_grad(const Vector& gamma, Vector& grad) const;

    // Compute the objective function, gradient, and T
    double dual_obj_grad(const Vector& gamma, Vector& grad, Matrix& T) const;

    // Compute the objective function, gradient, and the true Hessian
    void dual_obj_grad_densehess(
        const Vector& gamma, double& obj, Vector& grad, Matrix& hess
    ) const;

    // Compute the objective function, gradient, and sparsified Hessian
    void dual_obj_grad_hess(
        const Vector& gamma, double delta, double& obj, Vector& grad, Hessian& hess
    ) const;

    // Select step size
    double line_selection(
        const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
        double curobj, double& objval, bool verbose = false, std::ostream& cout = std::cout
    ) const;

    // Optimal beta given alpha
    void optimal_beta(const RefConstVec& alpha, RefVec beta) const;

    // Optimal alpha given beta
    void optimal_alpha(const RefConstVec& beta, RefVec alpha) const;
};

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_PROBLEM_H
