#include "qrot_hess.h"

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using ConstRefVec = Eigen::Ref<const Vector>;

namespace QROT {

// The generalized Hessian matrix has a special structure
//     H = [diag(sigma * 1m)              sigma]
//         [          sigma'  diag(sigma' * 1n)]
// sigma is an [n x m] matrix with all 0-1 values, computed from C [n x m]
//     sigma = ifelse(C >= 0, 1, 0)
//
// In our algorithm, we need to solve (H + lam * I)^{-1} * x
// Let h1 = sigma * 1m, h2 = sigma' * 1n,
//     Hl = [diag(h1) + lam * I               sigma]
//          [            sigma'  diag(h2) + lam * I]
//
// For x = [w, z], w [n], z [m], using the inverse formula of block matrices,
// we have
//     Hl^{-1} * x = [r1, r2]
//     r1 = v - (sigma * r2) ./ (h1 .+ lam)
//     r2 = Delta^{-1} * (z - y)
//     Delta = diag(h2) + lam * I - sigma' * diag(1 ./ (h1 .+ lam)) * sigma
//     v = w ./ (h1 .+ lam)  [=> r1 = (w - sigma * r2) ./ (h1 .+ lam)]
//     y = sigma' * v
//
// Since we primarily rely on conjugate gradient method to compute Delta^{-1} * y,
// we only need to implement the operator y -> Delta*y
//     Delta * y = h2 .* y + lam * y - sigma' * ((sigma * y) ./ (h1 .+ lam))
//
// Therefore, to represent H, we need to store two vector h1 and h2,
// and an efficient structure to compute sigma * z and sigma' * w

// Input gamma and M, compute the Hessian representations
//
// Also compute ||D||_F^2, D row sum, D column sum
// D[i, j] = (alpha[i] + beta[j] - M[i, j])+, gamma = [alpha, beta]
// Assume the memories of D_rowsum and D_colsum have been properly allocated
void Hessian::compute_D_hess(
    const Vector& gamma, const Matrix& M,
    double& D_sqnorm, double* D_rowsum, double* D_colsum
)
{
    // Initialization
    const int n = M.rows();
    const int m = M.cols();
    reset(n, m);
    D_sqnorm = 0.0;
    std::fill_n(D_rowsum, n, 0.0);
    std::fill_n(D_colsum, m, 0.0);

    // Pointers to h1 = sigma * 1m and h2 = sigma * 1n
    double* h1 = m_h1.data();
    double* h2 = m_h2.data();
    // Iterate over each element of D
    const double* alpha = gamma.data();
    const double* beta = alpha + n;
    const double* Mdata = M.data();
    for (int j = 0; j < m; j++)
    {
        const double betaj = beta[j];
        double colsumj = 0.0;
        int h2j = 0;
        for (int i = 0; i < n; i++, Mdata++)
        {
            const double Cij = alpha[i] + betaj - *Mdata;
            if (Cij > 0.0)
            {
                // C[i, j] > 0 => sigma[i, j] = 1
                //
                // 1. Add j to m_sigma[i]
                // 2. Add i to m_sigmat[j]
                // 3. Add D[i, j]^2 to D_sqnorm
                // 4. Add D[i, j] to D_rowsum[i]
                // 5. Add D[i, j] to D_colsum[j]
                m_sigma[i].push_back(j);
                m_sigmat[j].push_back(i);
                D_sqnorm += Cij * Cij;
                D_rowsum[i] += Cij;
                colsumj += Cij;

                // Let h1 = sigma * 1m, h2 = sigma * 1n
                //
                // 1. Add 1 to h1[i]
                // 2. Add 1 to h2[j]
                h1[i] += 1;
                h2j++;
            }
            // C[i, j] <= 0 => D[i, j] = 0
        }
        D_colsum[j] = colsumj;
        h2[j] = h2j;
    }
}

// res = sigma * x, x [m], res [n]
void Hessian::apply_sigmax(const double* x, double* res) const
{
    for (int i = 0; i < m_n; i++)
    {
        double resi = 0.0;
        for (auto j: m_sigma[i])
        {
            resi += x[j];
        }
        res[i] = resi;
    }
}

// res = sigma' * x, x [n], res [m]
void Hessian::apply_sigmatx(const double* x, double* res) const
{
    for (int j = 0; j < m_m; j++)
    {
        double resj = 0.0;
        for (auto i: m_sigmat[j])
        {
            resj += x[i];
        }
        res[j] = resj;
    }
}

// Compute A^(-1) * x
void Hessian::solve_Ax(const ConstRefVec& x, double shift, double tau, Vector& res) const
{
    res.resizeLike(x);

    // res is of size n, used as cache for h1_shift
    res.array() = m_h1.array() + shift;
    if (tau > 0.0)
    {
        Eigen::ArrayXd xh = x.array() / res.array();
        const double c1 = 1.0 / tau + (1.0 / res.array()).sum();
        const double c2 = xh.sum();
        res.array() = xh - (c2 / c1) / res.array();
    } else {
        res.array() = x.array() / res.array();
    }
}

// Compute B * x
void Hessian::apply_Bx(const ConstRefVec& x, double tau, Vector& res) const
{
    res.resize(m_n);

    // Compute sigma * x
    apply_sigmax(x.data(), res.data());

    if (tau > 0.0)
    {
        res.array() -= tau * x.sum();
    }
}

// Compute C * x
void Hessian::apply_Cx(const ConstRefVec& x, double tau, Vector& res) const
{
    res.resize(m_m);

    // Compute sigma * x
    apply_sigmatx(x.data(), res.data());

    if (tau > 0.0)
    {
        res.array() -= tau * x.sum();
    }
}

// Compute Delta * x
void Hessian::apply_Deltax(const Vector& x, double shift, double tau, Vector& res) const
{
    res.resizeLike(x);

    // Compute B * x
    apply_Bx(x, tau, m_cache_sigmax);
    // Compute A^(-1) * (B * x)
    solve_Ax(m_cache_sigmax, shift, tau, res);
    // Compute C * A^(-1) * (B * x)
    apply_Cx(res, tau, m_cache_sigmatx);
    // Compute (D - C * A^(-1) * B) * x
    res.noalias() = m_h2.cwiseProduct(x) + shift * x - m_cache_sigmatx;
    if (tau > 0.0)
    {
        res.array() += tau * x.sum();
    }
}

// Compute (H + shift * I + tau * K) * x
// K = vv', v = (1n, -1m)
// K * x = (v'x) v = (c * 1n, -c * 1m), c = x[:n].sum() - x[n:].sum()
void Hessian::apply_Hx(const Vector& x, double shift, double tau, Vector& res) const
{
    // x = [w, z], w [n], z [m]
    // H * x = [h1, h2] .* x + [sigma * z, sigma' * w]
    res.resizeLike(x);

    // Compute sigma * z
    apply_sigmax(x.data() + m_n, res.data());
    // Compute sigma' * w
    apply_sigmatx(x.data(), res.data() + m_n);
    // Add [h1, h2] .* x
    res.head(m_n).noalias() += m_h1.cwiseProduct(x.head(m_n));
    res.tail(m_m).noalias() += m_h2.cwiseProduct(x.tail(m_m));
    // Add shift * x
    if (shift != 0.0)
        res.noalias() += shift * x;
    if (tau > 0.0)
    {
        const double c = x.head(m_n).sum() - x.tail(m_m).sum();
        res.head(m_n).array() += tau * c;
        res.tail(m_m).array() -= tau * c;
    }
}

}  // namespace QROT
