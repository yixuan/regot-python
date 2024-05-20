#include <chrono>
#include "sinkhorn_hess.h"

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

inline double select_small_thresh(double* data, int n, double delta, bool inplace)
{
    // Get the sorted values
    Vector sorted;
    double* sorted_data;
    if (inplace)
    {
        sorted_data = data;
    } else {
        sorted.resize(n);
        std::copy(data, data + n, sorted.data());
        sorted_data = sorted.data();
    }
    std::sort(sorted_data, sorted_data + n);

    // Compute cumsum until the value is greater than delta
    double cumsum = 0.0, thresh = 0.0;
    for (int i = 0; i < n; i++, sorted_data++)
    {
        cumsum += *sorted_data;
        if (cumsum > delta)
            break;
        thresh = *sorted_data;
    }
    return thresh;
}

inline void apply_thresh_mask(double* data, int n, double thresh)
{
    for (int i = 0; i < n; i++, data++)
    {
        if (*data > thresh)
            *data = 0.0;
    }
}

inline void apply_thresh_mask(double* data, int n, int stride, double thresh)
{
    for (int i = 0; i < n; i++, data += stride)
    {
        if (*data > thresh)
            *data = 0.0;
    }
}

// Input gamma and M, compute the sparsified Hessian representations
void Hessian::compute_hess(Matrix& T, double reg, double delta)
{
    // Initialization
    const int n = T.rows();
    const int m = T.cols();
    reset(n, m);

    // Compute r = T * 1m, c = T' * 1n
    // h1 = r / reg, h2 = ct / reg
    m_h1.noalias() = T.rowwise().sum() / reg;
    m_h2.noalias() = T.leftCols(m_m - 1).colwise().sum().transpose() / reg;

    // Thresholding T by columns
    Matrix Delta = T;
    Delta.col(m_m - 1).setZero();
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < m_m - 1; j++)
    {
        int offset = m_n * j;
        double thresh = select_small_thresh(T.data() + offset, m_n, delta, false);
        apply_thresh_mask(Delta.data() + offset, m_n, thresh);
    }
    // Thresholding Delta by rows
    // Test row sums
    Vector rowsum = Delta.rowwise().sum();
    for (int i = 0; i < m_n; i++)
    {
        // For simplicity of theoretical analysis we use > delta,
        // but it is OK to multiply delta with a constant to save computation
        if (rowsum[i] > 2.0 * delta)
        {
            Vector row = Delta.row(i).transpose();
            double thresh = select_small_thresh(row.data(), m_m - 1, delta, true);
            apply_thresh_mask(Delta.data() + i, m_m - 1, m_n, thresh);
        }
    }
    // Generate sparse matrix
    m_sigmad = (T - Delta).leftCols(m_m - 1).sparseView();
    m_sigmad /= reg;
}

void Hessian::compute_hess2(Matrix& T, double reg, double delta)
{
    // Initialization
    const int n = T.rows();
    const int m = T.cols();
    reset(n, m);

    // Compute r = T * 1m, c = T' * 1n
    // h1 = r / reg, h2 = ct / reg
    m_h1.noalias() = T.rowwise().sum() / reg;
    m_h2.noalias() = T.leftCols(m_m - 1).colwise().sum().transpose() / reg;

    // Thresholding T by columns
    Matrix Delta1 = T;
    Delta1.col(m_m - 1).setZero();
    for (int j = 0; j < m_m - 1; j++)
    {
        int offset = m_n * j;
        double thresh = select_small_thresh(T.data() + offset, m_n, delta, false);
        apply_thresh_mask(Delta1.data() + offset, m_n, thresh);
    }
    // Thresholding T by rows
    Matrix Delta2 = T;
    Delta2.col(m_m - 1).setZero();
    for (int i = 0; i < m_n; i++)
    {
        Vector row = Delta2.row(i).transpose();
        double thresh = select_small_thresh(row.data(), m_m - 1, delta, false);
        apply_thresh_mask(Delta2.data() + i, m_m - 1, m_n, thresh);
    }
    // Generate sparse matrix
    m_sigmad = (T - Delta1.cwiseMin(Delta2)).leftCols(m_m - 1).sparseView();
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
