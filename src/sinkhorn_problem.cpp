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
    // res.array() = (data - c.replicate(1, m)).array().exp().rowwise().sum();
    // res.array() = c.array() + res.array().log();
    res.setZero();
    const double* cend = c.data() + n;
    for (int j = 0; j < m; j++)
    {
        const int offset = j * n;
        const double* cdata = c.data();
        double *rdata = res.data();
        const double* xdata = data.data() + offset;
        for (; cdata < cend; cdata++, rdata++, xdata++)
        {
            *rdata += std::exp((*xdata) - (*cdata));
        }
    }
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

// Compute T = exp((alpha (+) beta - M) / reg)
void Problem::compute_T(const Vector& gamma, Matrix& T) const
{
    T.resize(m_n, m_m);

    // Extract betat and set beta=(betat, 0)
    Vector beta(m_m);
    beta.head(m_m - 1).noalias() = gamma.tail(m_m - 1);
    beta[m_m - 1] = 0.0;

    // Compute T
    // T.noalias() = (gamma.head(m_n).replicate(1, m_m) +
    //     beta.transpose().replicate(m_n, 1) - m_M) / m_reg;
    // T.array() = T.array().exp();

#ifdef __AVX2__
    // Packet type
    using Scalar = double;
    using Eigen::internal::ploadu;
    using Eigen::internal::pstoreu;
    using Eigen::internal::pset1;
    using Eigen::internal::pexp;
    using Packet = Eigen::internal::packet_traits<Scalar>::type;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Compute for loop end points
    // n % (2^k) == n & (2^k-1), see https://stackoverflow.com/q/3072665
    // const int peeling_end = m_n - m_n % Increment;
    const int aligned_end = m_n - (m_n & (PacketSize - 1));
    const int peeling_end = m_n - (m_n & (Increment - 1));

    // Vectorized scalars
    const Packet vreg = pset1<Packet>(m_reg);

    for (int j = 0; j < m_m; j++)
    {
        const int offset = j * m_n;
        const double* alpha_head = gamma.data();
        const double* M_head = m_M.data() + offset;
        double* T_head = T.data() + offset;

        const double* alpha_data = alpha_head;
        const double* M_data = M_head;
        double* T_data = T_head;

        // Vectorized scalars
        const double betaj = beta[j];
        const Packet vbetaj = pset1<Packet>(betaj);

        for (int i = 0; i < peeling_end; i += Increment)
        {
            Packet valpha1 = ploadu<Packet>(alpha_data);
            Packet valpha2 = ploadu<Packet>(alpha_data + PacketSize);
            Packet vM1 = ploadu<Packet>(M_data);
            Packet vM2 = ploadu<Packet>(M_data + PacketSize);

            Packet vT1 = pexp((valpha1 + vbetaj - vM1) / vreg);
            Packet vT2 = pexp((valpha2 + vbetaj - vM2) / vreg);

            pstoreu(T_data, vT1);
            pstoreu(T_data + PacketSize, vT2);

            alpha_data += Increment;
            M_data += Increment;
            T_data += Increment;
        }
        if (aligned_end != peeling_end)
        {
            alpha_data = alpha_head + peeling_end;
            M_data = M_head + peeling_end;
            T_data = T_head + peeling_end;

            Packet valpha = ploadu<Packet>(alpha_data);
            Packet vM = ploadu<Packet>(M_data);
            Packet vT = pexp((valpha + vbetaj - vM) / vreg);

            pstoreu(T_data, vT);
        }
        // Remaining elements
        for (int i = aligned_end; i < m_n; i++)
        {
            T_head[i] = std::exp((alpha_head[i] + betaj - M_head[i]) / m_reg);
        }
    }
#else
    for (int j = 0; j < m_m; j++)
    {
        const double betaj = beta[j];
        const int offset = j * m_n;
        const double* alpha_data = gamma.data();
        const double* M_data = m_M.data() + offset;
        double* T_data = T.data() + offset;
        double* T_end = T_data + m_n;
        for (; T_data < T_end; T_data++, M_data++, alpha_data++)
        {
            *T_data = std::exp((*alpha_data + betaj - *M_data) / m_reg);
        }
    }
#endif
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
    // Call the second version below
    Matrix T(m_n, m_m);
    return dual_obj(gamma, T);
}

// Compute the objective function and T
double Problem::dual_obj(const Vector& gamma, Matrix& T) const
{
    // Compute T = exp((alpha (+) beta - M) / reg)
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
    grad.tail(m_m - 1).noalias() = T.leftCols(m_m - 1).colwise().sum().transpose() - m_b.head(m_m - 1);
}

// Compute the objective function and gradient
double Problem::dual_obj_grad(const Vector& gamma, Vector& grad) const
{
    // Call the second version below
    Matrix T(m_n, m_m);
    return dual_obj_grad(gamma, grad, T, true);
}

// Compute the objective function, gradient, and T
double Problem::dual_obj_grad(const Vector& gamma, Vector& grad, Matrix& T, bool computeT) const
{
    // Compute T = exp((alpha (+) beta - M) / reg)
    if (computeT)
        compute_T(gamma, T);

    grad.resize(m_n + m_m - 1);
    // g(alpha) = T * 1m - a
    grad.head(m_n).noalias() = T.rowwise().sum();
    double Tsum = grad.head(m_n).sum();
    grad.head(m_n).array() -= m_a.array();
    // g(beta) = T' * 1n - b
    grad.tail(m_m - 1).noalias() = T.leftCols(m_m - 1).colwise().sum().transpose() - m_b.head(m_m - 1);

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
    obj = dual_obj_grad(gamma, grad, T, true);

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
// void Problem::dual_obj_grad_hess(
//     const Vector& gamma, double delta, double& obj, Vector& grad, Hessian& hess
// ) const
// {
//     // Compute obj, grad, and T = exp((alpha (+) beta - M) / reg)
//     Matrix T(m_n, m_m);
//     obj = dual_obj_grad(gamma, grad, T, true);
//     // Compute sparsified Hessian from T
//     hess.compute_hess(T, grad, m_reg, delta, 0.001);
// }

void Problem::dual_sparsified_hess(
    const Matrix& T, const Vector& grad, double delta, double density_hint, Hessian& hess
) const
{
    // Row sums and column sums of T can be obtained from grad,
    // which saves some computation
    // r = T * 1m = grad_a + a, c = T' * 1n = grad_b + b
    Vector r = grad.head(m_n) + m_a;
    Vector ct = grad.tail(m_m - 1) + m_b.head(m_m - 1);

    hess.compute_hess(T, r, ct, m_reg, delta, density_hint);
}

// Select a step size
double Problem::line_selection(
    const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
    double curobj, Matrix& T, double& objval, bool verbose,
    std::ostream& cout
) const
{
    const int nc = static_cast<int>(candid.size());
    double best_step = 1.0;
    objval = std::numeric_limits<double>::infinity();
    for (int i = 0; i < nc; i++)
    {
        const double alpha = candid[i];
        Vector newgamma = gamma + alpha * direc;
        const double objfn = dual_obj(newgamma, T);
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
