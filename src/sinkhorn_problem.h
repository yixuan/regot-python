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

    // Compute T = exp((alpha (+) beta - M) / reg) and return sum(T)
    double compute_T(const Vector& gamma, Matrix& T) const;

    // Save row sums to Tsums[0:n], and
    // save column sums (excluding the last column) to Tsums[n:(n + m - 1)]
    void compute_sums(const Matrix& T, Vector& Tsums) const;

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

    // Compute the objective function and T
    double dual_obj(const Vector& gamma, Matrix& T) const;

    // Compute the gradient
    void dual_grad(const Vector& gamma, Vector& grad) const;

    // Compute the objective function and gradient
    double dual_obj_grad(const Vector& gamma, Vector& grad) const;

    // Compute the objective function, gradient, and T
    double dual_obj_grad(const Vector& gamma, Vector& grad, Matrix& T, bool computeT) const;

    // Compute the objective function, gradient, and the true Hessian
    void dual_obj_grad_densehess(
        const Vector& gamma, double& obj, Vector& grad, Matrix& hess
    ) const;

    /*
    Compute the objective function, gradient,
    and the true sparsified Hessian in dense format.
    Finally hess = hess + shift * I to avoid singularity.
    */
    void dual_obj_grad_sparsehess_dense(
        const Vector& gamma, double& obj, Vector& grad, Matrix& hess, double density, double shift
    ) const;

    // Compute the objective function, gradient, and sparsified Hessian
    // void dual_obj_grad_hess(
    //     const Vector& gamma, double delta, double& obj, Vector& grad, Hessian& hess
    // ) const;
    void dual_sparsified_hess(
        const Matrix& T, const Vector& grad, double delta, double density_hint, Hessian& hess
    ) const;

    // Compute the sparsified Hessian with density specified
    void dual_sparsified_hess_with_density(
        const Matrix& T, const Vector& grad, double density, Hessian& hess
    ) const;

    // Select step size
    double line_selection(
        const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
        double curobj, Matrix& T, double& objval, bool verbose = false,
        std::ostream& cout = std::cout
    ) const;

    /*
    Backtracking line search with wolfe conditions.
    */
    double line_search_backtracking(
        const Vector& gamma, const Vector& direc,
        double f, const Vector& g,
        double c1 = 1e-4, double c2 = 0.9,
        int max_iter = 20, bool verbose = false,
        std::ostream& cout = std::cout
    ) const;

    // Optimal beta given alpha
    void optimal_beta(const RefConstVec& alpha, RefVec beta) const;

    // Optimal alpha given beta
    void optimal_alpha(const RefConstVec& beta, RefVec alpha) const;

};

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_PROBLEM_H
