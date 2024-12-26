#include "sinkhorn_problem.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefVec = Eigen::Ref<Vector>;
using RefConstVec = Eigen::Ref<const Vector>;
using RefMat = Eigen::Ref<Matrix>;
using RefConstMat = Eigen::Ref<const Matrix>;

// log(exp(x[0]) + exp(x[1]) + ... + exp(x[n-1]))
// = c + log(exp(x[0] - c) + ... + exp(x[n-1] - c))
inline double log_sum_exp(const double* data, const int n)
{
    // Get the maximum element in the array
    const double c = *(std::max_element(data, data + n));
    double res = 0.0;

#ifdef __AVX2__
    // Packet type
    using Scalar = double;
    using Eigen::internal::ploadu;
    using Eigen::internal::pset1;
    using Eigen::internal::padd;
    using Eigen::internal::psub;
    using Eigen::internal::pexp;
    using Eigen::internal::predux;
    using Packet = Eigen::internal::packet_traits<Scalar>::type;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Vectorized scalars
    const Packet vc = pset1<Packet>(c);

    // Compute for loop end points
    // n % (2^k) == n & (2^k-1), see https://stackoverflow.com/q/3072665
    // const Index peeling_end = n - n % Increment;
    const int aligned_end = n - (n & (PacketSize - 1));
    const int peeling_end = n - (n & (Increment - 1));

    // Working pointers
    const double* xdata = data;
    Packet vres = pset1<Packet>(0);
    for (int i = 0; i < peeling_end; i += Increment)
    {
        // Load data
        Packet vx1 = ploadu<Packet>(xdata);
        Packet vx2 = ploadu<Packet>(xdata + PacketSize);
        // Compute result
        Packet vres1 = pexp(psub(vx1, vc));
        Packet vres2 = pexp(psub(vx2, vc));
        // Reduce
        vres = padd(vres, padd(vres1, vres2));

        xdata += Increment;
    }
    if (aligned_end != peeling_end)
    {
        xdata = data + peeling_end;

        Packet vx = ploadu<Packet>(xdata);
        vres = padd(vres, pexp(psub(vx, vc)));
    }
    // Reduce to scalar
    res = predux(vres);
    // Remaining elements
    for (int i = aligned_end; i < n; i++)
    {
        res += std::exp(data[i] - c);
    }
#else
    for (int i = 0; i < n; i++, data++)
        res += std::exp(*data - c);
#endif

    return c + std::log(res);
}

// data [n x m], res [n]
inline void log_sum_exp_rowwise(const RefConstMat& data, RefVec res)
{
    const int n = data.rows();
    const int m = data.cols();
    res.resize(n);
    Vector c = data.rowwise().maxCoeff();

    // // Implementation 1
    // res.array() = (data - c.replicate(1, m)).array().exp().rowwise().sum();
    // res.array() = c.array() + res.array().log();

    // Implementation 2
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

// data [n x m], res [m]
inline void log_sum_exp_colwise(const RefConstMat& data, RefVec res)
{
    const int n = data.rows();
    const int m = data.cols();
    res.resize(m);

    // Implementation 1
    Vector c = data.colwise().maxCoeff();
    res.array() = (data - c.transpose().replicate(n, 1)).array().exp().colwise().sum();
    res.array() = c.array() + res.array().log();

    // // Implementation 2
    // Vector c = data.colwise().maxCoeff();
    // for (int j = 0; j < m; j++)
    // {
    //     double rj = 0.0;
    //     const double cj = c[j];
    //     const int offset = j * n;
    //     const double* xdata = data.data() + offset;
    //     const double* xend = xdata + n;
    //     for (; xdata < xend; xdata++)
    //     {
    //         rj += std::exp((*xdata) - cj);
    //     }
    //     res[j] = rj;
    // }
    // res.array() = c.array() + res.array().log();

    // // Implementation 3
    // for (int j = 0; j < m; j++)
    // {
    //     const int offset = j * n;
    //     const double* xdata = data.data() + offset;
    //     res[j] = log_sum_exp(xdata, n);
    // }
}

// Optimal beta given alpha
void Problem::optimal_beta(const RefConstVec& alpha, RefVec beta) const
{
    beta.resize(m_m);
    Matrix D(m_n, m_m);

    // Impelementation 1
    // D.array() = (alpha.replicate(1, m_m) - m_M).array() / m_reg;

    // Implementation 2
    const double* alpha_head = alpha.data();
    const double* alpha_end = alpha_head + m_n;
    for (int j = 0; j < m_m; j++)
    {
        const int offset = j * m_n;
        const double* adata = alpha_head;
        const double* Mdata = m_M.data() + offset;
        double* Ddata = D.data() + offset;
        for (; adata < alpha_end; adata++, Mdata++, Ddata++)
        {
            *Ddata = (*adata - *Mdata) / m_reg;
        }
    }

    log_sum_exp_colwise(D, beta);
    beta.noalias() = m_reg * (m_logb - beta);
}

// Optimal alpha given beta
void Problem::optimal_alpha(const RefConstVec& beta, RefVec alpha) const
{
    alpha.resize(m_n);
    Matrix D(m_n, m_m);

    // Impelementation 1
    D.array() = (beta.transpose().replicate(m_n, 1) - m_M).array() / m_reg;

    // // Implementation 2
    // for (int j = 0; j < m_m; j++)
    // {
    //     const double betaj = beta[j];
    //     const int offset = j * m_n;
    //     const double* Mdata = m_M.data() + offset;
    //     const double* Mend = Mdata + m_n;
    //     double* Ddata = D.data() + offset;
    //     for (; Mdata < Mend; Mdata++, Ddata++)
    //     {
    //         *Ddata = (betaj - *Mdata) / m_reg;
    //     }
    // }

    log_sum_exp_rowwise(D, alpha);
    alpha.noalias() = m_reg * (m_loga - alpha);
}

}  // namespace Sinkhorn
