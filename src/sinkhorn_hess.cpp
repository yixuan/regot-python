#include <chrono>
#include "sinkhorn_hess.h"
#include "sinkhorn_sparsify.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefVec = Eigen::Ref<Vector>;
using ConstRefVec = Eigen::Ref<const Vector>;
using MapVec = Eigen::Map<Vector>;
using ConstMapVec = Eigen::Map<const Vector>;
using SpMat = Eigen::SparseMatrix<double>;
// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

// Input gamma and M, compute the sparsified Hessian representations
void Hessian::compute_hess(
    const Matrix& T, const Vector& Trowsum, const Vector& Ttcolsum,
    double reg, double delta, double density_hint
)
{
    // Initialization
    const int n = T.rows();
    const int m = T.cols();
    reset(n, m);

    // Compute r = T * 1m, c = T' * 1n
    // h1 = r / reg, h2 = ct / reg
    m_h1.noalias() = Trowsum / reg;
    m_h2.noalias() = Ttcolsum / reg;

    // Sparsify T
    m_sigmad = sparsify_mat3(T, delta, density_hint);
    m_sigmad /= reg;
}

SpMat Hessian::to_spmat(bool only_lower) const
{
    std::vector<Eigen::Triplet<double>> coeffs;
    // Diagonal elements
    for (int i = 0; i < m_n; i++)
        coeffs.emplace_back(i, i, m_h1[i]);
    for (int i = 0; i < m_m - 1; i++)
        coeffs.emplace_back(m_n + i, m_n + i, m_h2[i]);
    // Off-diagonal elements
    const int outer_size = m_sigmad.outerSize();
    if (only_lower)
    {
        for (int k = 0; k < outer_size; k++)
        {
            for (SpMat::InnerIterator it(m_sigmad, k); it; ++it)
            {
                const int i = it.row();
                const int j = it.col();
                const double val = it.value();
                coeffs.emplace_back(m_n + j, i, val);
            }
        }
    } else {
        for (int k = 0; k < outer_size; k++)
        {
            for (SpMat::InnerIterator it(m_sigmad, k); it; ++it)
            {
                const int i = it.row();
                const int j = it.col();
                const double val = it.value();
                coeffs.emplace_back(i, m_n + j, val);
                coeffs.emplace_back(m_n + j, i, val);
            }
        }
    }

    SpMat H(m_n + m_m - 1, m_n + m_m - 1);
    H.setFromTriplets(coeffs.begin(), coeffs.end());
    return H;
}

// res = sigmad * x, x [m-1], res [n]
void Hessian::apply_sigmadx(const double* x, double* res) const
{
    ConstMapVec xm(x, m_m - 1);
    MapVec rm(res, m_n);
    rm.noalias() = m_sigmad * xm;
}

// res = sigmad' * x, x [n], res [m-1]
void Hessian::apply_sigmadtx(const double* x, double* res) const
{
    ConstMapVec xm(x, m_n);
    MapVec rm(res, m_m - 1);
    rm.noalias() = m_sigmad.transpose() * xm;
}

// Compute A^(-1) * x
void Hessian::solve_Ax(const ConstRefVec& x, double shift, Vector& res) const
{
    res.resizeLike(x);
    res.array() = x.array() / (m_h1.array() + shift);
}

// Compute B * x
void Hessian::apply_Bx(const ConstRefVec& x, Vector& res) const
{
    res.resize(m_n);
    apply_sigmadx(x.data(), res.data());
}

// Compute C * x
void Hessian::apply_Cx(const ConstRefVec& x, Vector& res) const
{
    res.resize(m_m - 1);
    apply_sigmadtx(x.data(), res.data());
}

// Compute Delta * x
void Hessian::apply_Deltax(const Vector& x, double shift, Vector& res) const
{
    res.resizeLike(x);

    // Compute B * x
    apply_Bx(x, m_cache_sigmadx);
    // Compute A^(-1) * (B * x)
    solve_Ax(m_cache_sigmadx, shift, res);
    // Compute C * A^(-1) * (B * x)
    apply_Cx(res, m_cache_sigmadtx);
    // Compute (D - C * A^(-1) * B) * x
    res.noalias() = m_h2.cwiseProduct(x) + shift * x - m_cache_sigmadtx;
}

// Compute Hs * x
void Hessian::apply_Hsx(const Vector& x, Vector& res) const
{
    // x = [w, z], w [n], z [m-1]
    // Hs * x = [h1, h2] .* x + [sigmad * z, sigmad' * w]
    res.resizeLike(x);

    // Compute sigmad * z
    apply_sigmadx(x.data() + m_n, res.data());
    // Compute sigmad' * w
    apply_sigmadtx(x.data(), res.data() + m_n);
    // Add [h1, h2] .* x
    res.head(m_n).noalias() += m_h1.cwiseProduct(x.head(m_n));
    res.tail(m_m - 1).noalias() += m_h2.cwiseProduct(x.tail(m_m - 1));
}

}  // namespace Sinkhorn
