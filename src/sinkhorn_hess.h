#ifndef REGOT_SINKHORN_HESS_H
#define REGOT_SINKHORN_HESS_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace Sinkhorn {

// The Hessian matrix has the following form
//     H = [diag(h1)     sigma]
//         [  sigma'  diag(h2)]
//
// h1 = r / reg, h2 = ct / reg, sigma = Tt / reg
// T is an [n x m] matrix, T = exp((alpha (+) beta - M) / reg)
// Tt contains the first (m-1) columns of T
// r = T * 1m, c = T' * 1n, ct contains the first (m-1) elements of c
//
// The sparsified Hessian matrix has the following form
//     Hs = [diag(h1)    sigmad]
//          [ sigmad'  diag(h2)]
// sigmad is a sparsified version of sigma
//
// In our algorithm, we need to solve
//     (Hs + shift * I)^{-1} * x
// Define Hl = Hs + shift * I
//     Hl = [diag(h1) + shift * I                sigmad]
//          [             sigmad'  diag(h2) + shift * I]
//
// For x = [w, z], w [n], z [m-1], using the inverse formula of block matrices,
// we have
//     Hl^{-1} * x = [r1, r2]
//     r1 = v - (sigmad * r2) ./ (h1 .+ shift)
//     r2 = Delta^{-1} * (z - y)
//     Delta = diag(h2) + shift * I - sigmad' * diag(1 ./ (h1 .+ shift)) * Tdt
//     v = w ./ (r .+ shift)  [=> r1 = (w - sigmad * r2) ./ (h1 .+ shift)]
//     y = sigmad' * v
//
// Since we primarily rely on conjugate gradient method to compute Delta^{-1} * y,
// we only need to implement the operator y -> Delta*y
//     Delta * y = h2 .* y + shift * y - sigmad' * ((sigmad * y) ./ (h1 .+ shift))
//
// Therefore, to represent Hs, we need to store two vector h1 and h2,
// and a sparse matrix sigmad
class Hessian
{
private:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    using SpMat = Eigen::SparseMatrix<double>;
    using ConstRefVec = Eigen::Ref<const Vector>;
    using RefConstMat = Eigen::Ref<const Matrix>;

    int m_n;
    int m_m;

    // T * 1m / reg, [n] vector
    Vector m_h1;
    // Tt' * 1n / reg, [m-1] vector
    Vector m_h2;
    // Sparsified Tt/reg, [n x (m-1)] sparse matrix
    SpMat  m_sigmad;

    // Caches
    mutable Vector m_cache_sigmadx;
    mutable Vector m_cache_sigmadtx;

public:
    int size_n() const { return m_n; }
    int size_m() const { return m_m; }
    const Vector& h1() const { return m_h1; }
    const Vector& h2() const { return m_h2; }
    const SpMat& sigmad() const { return m_sigmad; }
    double density() const
    {
        int sigmad_nnz = m_sigmad.nonZeros();
        int nnz = 2 * sigmad_nnz + m_n + m_m - 1;
        return double(nnz) / double(m_n + m_m - 1) / double(m_n + m_m - 1);
    }

    // Swap with another object
    inline void swap(Hessian& other)
    {
        std::swap(this->m_n, other.m_n);
        std::swap(this->m_m, other.m_m);
        this->m_h1.swap(other.m_h1);
        this->m_h2.swap(other.m_h2);
        this->m_sigmad.swap(other.m_sigmad);
        this->m_cache_sigmadx.swap(other.m_cache_sigmadx);
        this->m_cache_sigmadtx.swap(other.m_cache_sigmadtx);
    }

    // For debugging purpose
    inline void print(std::ostream& cout = std::cout) const
    {
        cout << "h1 = " << m_h1.transpose() << std::endl;
        cout << "h2 = " << m_h2.transpose() << std::endl;
        cout << "sigmad = " << m_sigmad << std::endl;
    }

    // Initialization and resetting
    inline void reset(int n, int m)
    {
        m_n = n;
        m_m = m;
        m_h1.resize(n);
        m_h1.setZero();
        m_h2.resize(m - 1);
        m_h2.setZero();
        m_sigmad.resize(n, m - 1);
        m_cache_sigmadx.resize(n);
        m_cache_sigmadtx.resize(m - 1);
    }

    // Input T, compute the sparsified Hessian representations
    void compute_hess(Matrix& T, double reg, double delta);
    void compute_hess2(Matrix& T, double reg, double delta);

    // Convert to an Eigen sparse matrix
    SpMat to_spmat(bool only_lower=true) const;

    // res = sigmad * x, x [m-1], res [n]
    void apply_sigmadx(const double* x, double* res) const;

    // res = sigmad' * x, x [n], res [m-1]
    void apply_sigmadtx(const double* x, double* res) const;

    // [A  B], Delta = (D - C * A^(-1) * B)
    // [C  D]

    // Compute A^(-1) * x
    void solve_Ax(const ConstRefVec& x, double shift, Vector& res) const;

    // Compute B * x
    void apply_Bx(const ConstRefVec& x, Vector& res) const;

    // Compute C * x
    void apply_Cx(const ConstRefVec& x, Vector& res) const;

    // Compute Delta * x
    // Delta * x = h2 .* x + shift * x - sigmad' * ((sigmad * x) ./ (h1 .+ shift))
    void apply_Deltax(const Vector& x, double shift, Vector& res) const;

    // Compute Hs * x
    void apply_Hsx(const Vector& x, Vector& res) const;
};

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_HESS_H
