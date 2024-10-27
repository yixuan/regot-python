#ifndef REGOT_QROT_PROBLEM_H
#define REGOT_QROT_PROBLEM_H

#include <Eigen/Core>
#include "qrot_hess.h"

namespace QROT {

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
    const double m_tau;

public:
    Problem(const RefConstMat& M, const RefConstVec& a, const RefConstVec& b, double reg, double tau):
        m_n(M.rows()), m_m(M.cols()),
        m_M(M), m_a(a), m_b(b),
        m_reg(reg), m_tau(tau)
    {}

    // Return the dimensions
    int size_n() const { return m_n; }
    int size_m() const { return m_m; }

    // Return the matrices/vectors
    RefConstMat get_M() const { return m_M; }
    RefConstVec get_a() const { return m_a; }
    RefConstVec get_b() const { return m_b; }

    // Return the regularization parameter
    double reg() const { return m_reg; }
    double tau() const { return m_tau; }

    // Compute the primal objective function
    double primal_val(const Vector& gamma) const;

    // Compute the objective function
    // f(alpha, beta) = 0.5 * ||(alpha (+) beta - M)+||^2
    //                    - reg * (a' * alpha + b' * beta)
    //                    + (tau / 2) * (1' * alpha - 1' * beta)^2
    double dual_obj_vanilla(const Vector& gamma) const;
    // SIMD version
    double dual_obj(const Vector& gamma) const;

    // Compute the gradient
    void dual_grad(const Vector& gamma, Vector& grad) const;

    // Compute the objective function and gradient
    double dual_obj_grad(const Vector& gamma, Vector& grad) const;

    // Compute the objective function, gradient, and generalized Hessian
    // C = alpha (+) beta - M, D = (C)+, sigma = ifelse(C >= 0, 1, 0)
    // f(alpha, beta) = 0.5 * ||D||^2 - reg * (a' * alpha + b' * beta)
    //                    + (tau / 2) * (1' * alpha - 1' * beta)^2
    // g(alpha) = D * 1m - reg * a + tau * (1' * alpha - 1' * beta) * 1n
    // g(beta) = D' * 1n - reg * b - tau * (1' * alpha - 1' * beta) * 1m
    // H = [diag(sigma * 1m)              sigma] + tau * K
    //     [          sigma'  diag(sigma' * 1n)]
    // K = vv', v = (1n, -1m)
    void dual_obj_grad_hess(
        const Vector& gamma, double& obj, Vector& grad, Hessian& hess
    ) const;

    // Select a step size
    double line_selection(
        const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
        double& objval, bool verbose = false, std::ostream& cout = std::cout
    ) const;
    double line_selection2(
        const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
        double curobj, double& objval, bool verbose = false, std::ostream& cout = std::cout
    ) const;

    // Optimal beta given alpha
    void optimal_beta(const RefConstVec& alpha, RefVec beta) const;

    // Optimal alpha given beta
    void optimal_alpha(const RefConstVec& beta, RefVec alpha) const;

    // Compute the objective function and gradient of semi-dual
    double semi_dual_obj_grad(const Vector& alpha, Vector& grad) const;
};

}  // namespace QROT


#endif  // REGOT_QROT_PROBLEM_H
