#include "sinkhorn_problem.h"
#include "approx_proj.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefVec = Eigen::Ref<Vector>;
using RefConstVec = Eigen::Ref<const Vector>;
using RefMat = Eigen::Ref<Matrix>;
using RefConstMat = Eigen::Ref<const Matrix>;

// T[i] = exp((alpha[i] + beta - M[i]) / reg)
// Return sum(T)
template <bool NoBeta = false>
double compute_column_helper(
    double* T, const double* alpha, const double* M,
    const double beta, const double reg,
    const int n, const int aligned_end, const int peeling_end
)
{
    // Working pointers
    const double* alpha_data = alpha;
    const double* M_data = M;
    double* T_data = T;

    // Sum
    double Tsum = 0.0;

#ifdef __AVX2__
    // Packet type
    using Scalar = double;
    using Eigen::internal::ploadu;
    using Eigen::internal::pstoreu;
    using Eigen::internal::pset1;
    using Eigen::internal::padd;
    using Eigen::internal::psub;
    using Eigen::internal::pdiv;
    using Eigen::internal::pexp;
    using Eigen::internal::predux;
    using Packet = Eigen::internal::packet_traits<Scalar>::type;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Vectorized scalars
    const Packet vreg = pset1<Packet>(reg);
    const Packet vbeta = pset1<Packet>(beta);

    // Packet sum
    Packet vTsum = pset1<Packet>(0);

    for (int i = 0; i < peeling_end; i += Increment)
    {
        // Load data
        Packet valpha1 = ploadu<Packet>(alpha_data);
        Packet valpha2 = ploadu<Packet>(alpha_data + PacketSize);
        Packet vM1 = ploadu<Packet>(M_data);
        Packet vM2 = ploadu<Packet>(M_data + PacketSize);

        // Compute T
        Packet vT1, vT2;
        if (NoBeta)
        {
            vT1 = pexp(pdiv(psub(valpha1, vM1), vreg));
            vT2 = pexp(pdiv(psub(valpha2, vM2), vreg));
        } else {
            vT1 = pexp(pdiv(psub(padd(valpha1, vbeta), vM1), vreg));
            vT2 = pexp(pdiv(psub(padd(valpha2, vbeta), vM2), vreg));
        }
        // Reduce
        vTsum = padd(vTsum, padd(vT1, vT2));

        // Store T
        pstoreu(T_data, vT1);
        pstoreu(T_data + PacketSize, vT2);

        // Increment pointers
        alpha_data += Increment;
        M_data += Increment;
        T_data += Increment;
    }
    if (aligned_end != peeling_end)
    {
        alpha_data = alpha + peeling_end;
        M_data = M + peeling_end;
        T_data = T + peeling_end;

        Packet valpha = ploadu<Packet>(alpha_data);
        Packet vM = ploadu<Packet>(M_data);
        Packet vT;
        if (NoBeta)
        {
            vT = pexp(pdiv(psub(valpha, vM), vreg));
        } else {
            vT = pexp(pdiv(psub(padd(valpha, vbeta), vM), vreg));
        }
        vTsum = padd(vTsum, vT);

        pstoreu(T_data, vT);
    }

    // Reduce to scalar
    Tsum += predux(vTsum);

    // Remaining elements
    for (int i = aligned_end; i < n; i++)
    {
        double Ti = 0.0;
        if (NoBeta)
        {
            Ti = std::exp((alpha[i] - M[i]) / reg);
        } else {
            Ti = std::exp((alpha[i] + beta - M[i]) / reg);
        }
        Tsum += Ti;
        T[i] = Ti;
    }
#else
    const double* alpha_end = alpha + n;
    for (; alpha_data < alpha_end; alpha_data++, M_data++, T_data++)
    {
        double Ti = 0.0;
        if (NoBeta)
        {
            Ti = std::exp((*alpha_data - *M_data) / reg);
        } else {
            Ti = std::exp((*alpha_data + beta - *M_data) / reg);
        }
        Tsum += Ti;
        *T_data = Ti;
    }
#endif

    return Tsum;
}

// Compute T = exp((alpha (+) beta - M) / reg) and return sum(T)
double Problem::compute_T(const Vector& gamma, Matrix& T) const
{
    T.resize(m_n, m_m);
    double Tsum = 0.0;

    // Compute T
    // T.noalias() = (gamma.head(m_n).replicate(1, m_m) +
    //     beta.transpose().replicate(m_n, 1) - m_M) / m_reg;
    // T.array() = T.array().exp();

#ifdef __AVX2__
    // Packet type
    using Scalar = double;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Compute for loop end points
    // n % (2^k) == n & (2^k-1), see https://stackoverflow.com/q/3072665
    // const int peeling_end = m_n - m_n % Increment;
    const int aligned_end = m_n - (m_n & (PacketSize - 1));
    const int peeling_end = m_n - (m_n & (Increment - 1));
#else
    const int aligned_end = m_n;
    const int peeling_end = m_n;
#endif

    // Working pointers
    const double* alpha = gamma.data();
    const double* beta = alpha + m_n;
    const double* M_head = m_M.data();
    double* T_head = T.data();

    // First (m-1) columns
    const int m1 = m_m - 1;
    for (int j = 0; j < m1; j++, T_head += m_n, M_head += m_n)
    {
        Tsum += compute_column_helper<false>(
            T_head, alpha, M_head, beta[j], m_reg,
            m_n, aligned_end, peeling_end
        );
    }
    // Last column
    Tsum += compute_column_helper<true>(
        T_head, alpha, M_head, 0.0, m_reg,
        m_n, aligned_end, peeling_end
    );

    return Tsum;
}

// Save row sums to Tsums[0:n], and
// save column sums (excluding the last column) to Tsums[n:(n + m - 1)]
void Problem::compute_sums(const Matrix& T, Vector& Tsums) const
{
    Tsums.resize(m_n + m_m - 1);
    Tsums.head(m_n).setZero();

    // Pointers to output row sums and column sums
    double* rs = Tsums.data();
    double* cs = rs + m_n;

#ifdef __AVX2__
    // Packet type
    using Scalar = double;
    using Eigen::internal::ploadu;
    using Eigen::internal::pstoreu;
    using Eigen::internal::pset1;
    using Eigen::internal::padd;
    using Eigen::internal::predux;
    using Packet = Eigen::internal::packet_traits<Scalar>::type;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Compute for loop end points
    // n % (2^k) == n & (2^k-1), see https://stackoverflow.com/q/3072665
    // const int peeling_end = m_n - m_n % Increment;
    const int aligned_end = m_n - (m_n & (PacketSize - 1));
    const int peeling_end = m_n - (m_n & (Increment - 1));

    for (int j = 0; j < m_m; j++)
    {
        const int offset = j * m_n;
        const double* T_head = T.data() + offset;
        const double* T_data = T_head;
        double* rs_data = rs;

        // Column sum
        Packet vTcolsumj = pset1<Packet>(0);
        double Tcolsumj = 0.0;

        for (int i = 0; i < peeling_end; i += Increment)
        {
            Packet vT1 = ploadu<Packet>(T_data);
            Packet vT2 = ploadu<Packet>(T_data + PacketSize);
            Packet vrs1 = ploadu<Packet>(rs_data);
            Packet vrs2 = ploadu<Packet>(rs_data + PacketSize);

            // For column sums
            vTcolsumj = padd(vTcolsumj, padd(vT1, vT2));

            // For row sums
            pstoreu(rs_data, padd(vrs1, vT1));
            pstoreu(rs_data + PacketSize, padd(vrs2, vT2));

            T_data += Increment;
            rs_data += Increment;
        }
        if (aligned_end != peeling_end)
        {
            T_data = T_head + peeling_end;
            rs_data = rs + peeling_end;

            Packet vT = ploadu<Packet>(T_data);
            Packet vrs = ploadu<Packet>(rs_data);

            // For column sums
            vTcolsumj = padd(vTcolsumj, vT);

            // For row sums
            pstoreu(rs_data, padd(vrs, vT));
        }
        Tcolsumj = predux(vTcolsumj);
        // Remaining elements
        for (int i = aligned_end; i < m_n; i++)
        {
            double Ti = T_head[i];
            Tcolsumj += Ti;
            rs[i] += Ti;
        }

        // Save column sum
        if (j < m_m - 1)
            cs[j] = Tcolsumj;

    }
#else
    for (int j = 0; j < m_m; j++)
    {
        double Tcolsumj = 0.0;

        const int offset = j * m_n;
        const double* T_data = T.data() + offset;
        const double* T_end = T_data + m_n;
        double* rs_data = rs;
        for (; T_data < T_end; T_data++, rs_data++)
        {
            double Ti = *T_data;
            Tcolsumj += Ti;
            *rs_data += Ti;
        }

        // Save column sum
        if (j < m_m - 1)
            cs[j] = Tcolsumj;
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
    double Tsum = compute_T(gamma, T);

    // Compute objective function value
    double obj = m_reg * Tsum - gamma.head(m_n).dot(m_a) -
        gamma.tail(m_m - 1).dot(m_b.head(m_m - 1));

    return obj;
}

// Compute the gradient
void Problem::dual_grad(const Vector& gamma, Vector& grad) const
{
    grad.resize(m_n + m_m - 1);

    // Compute T = exp((alpha (+) beta - M) / reg)
    Matrix T(m_n, m_m);
    compute_T(gamma, T);
    // Compute row sums and column sums of T
    compute_sums(T, grad);

    // Now grad stores T * 1m and T' * 1n (excluding the last column)
    // g(alpha) = T * 1m - a
    grad.head(m_n).noalias() -= m_a;
    // g(beta) = T' * 1n - b
    grad.tail(m_m - 1).noalias() -= m_b.head(m_m - 1);
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
    grad.resize(m_n + m_m - 1);

    // Compute T = exp((alpha (+) beta - M) / reg)
    if (computeT)
        compute_T(gamma, T);
    // Compute row sums and column sums of T
    compute_sums(T, grad);

    // Now grad stores T * 1m and T' * 1n (excluding the last column)
    double Tsum = grad.head(m_n).sum();
    // g(alpha) = T * 1m - a
    grad.head(m_n).noalias() -= m_a;
    // g(beta) = T' * 1n - b
    grad.tail(m_m - 1).noalias() -= m_b.head(m_m - 1);

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

    // Row sums and column sums of T can be obtained from grad,
    // which saves some computation
    // r = T * 1m = grad_a + a, c = T' * 1n = grad_b + b
    hess.diagonal().head(m_n).noalias() = grad.head(m_n) + m_a;
    hess.diagonal().tail(m_m - 1).noalias() = grad.tail(m_m - 1) + m_b.head(m_m - 1);

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

// Compute the sparsified Hessian with density specified
void Problem::dual_sparsified_hess_with_density(
    const Matrix& T, const Vector& grad, double density, Hessian& hess
) const
{
    // Row sums and column sums of T can be obtained from grad,
    // which saves some computation
    // r = T * 1m = grad_a + a, c = T' * 1n = grad_b + b
    Vector r = grad.head(m_n) + m_a;
    Vector ct = grad.tail(m_m - 1) + m_b.head(m_m - 1);

    hess.compute_hess_with_density(T, r, ct, m_reg, density);
}

}  // namespace Sinkhorn

