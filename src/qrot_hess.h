#ifndef REGOT_QROT_HESS_H
#define REGOT_QROT_HESS_H

#include <iostream>
#include <vector>
#include <Eigen/Core>

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
class Hessian
{
private:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    int m_n;
    int m_m;

    // sigma * 1m, [n] vector
    Vector m_h1;
    // sigma' * 1n, [m] vector
    Vector m_h2;
    // m_sigma represents the locations of ones in sigma
    // J(i) = {m_sigma[i]}, then sigma[i, J(i)] = 1
    std::vector<std::vector<int>> m_sigma;
    // m_sigmat also represents the locations of ones in sigma
    // I(j) = {m_sigmat[j]}, then sigma[I(j), j] = 1
    std::vector<std::vector<int>> m_sigmat;

    // Caches
    mutable Vector m_cache_sigmax;
    mutable Vector m_cache_sigmatx;

public:
    int size_n() const { return m_n; }
    int size_m() const { return m_m; }
    const Vector& h1() const { return m_h1; }
    const Vector& h2() const { return m_h2; }

    // Swap with another object
    inline void swap(Hessian& other)
    {
        std::swap(this->m_n, other.m_n);
        std::swap(this->m_m, other.m_m);
        this->m_h1.swap(other.m_h1);
        this->m_h2.swap(other.m_h2);
        this->m_sigma.swap(other.m_sigma);
        this->m_sigmat.swap(other.m_sigmat);
        this->m_cache_sigmax.swap(other.m_cache_sigmax);
        this->m_cache_sigmatx.swap(other.m_cache_sigmatx);
    }

    // For debugging purpose
    inline void print(std::ostream& cout = std::cout) const
    {
        cout << "h1 = " << m_h1.transpose() << std::endl;
        cout << "h2 = " << m_h2.transpose() << std::endl;
        for(std::size_t i = 0; i < m_sigma.size(); i++)
        {
            for(auto e: m_sigma[i])
            {
                cout << "(" << i << ", " << e << ")" << std::endl;
            }
        }
    }

    // Initialization and resetting
    inline void reset(int n, int m)
    {
        m_n = n;
        m_m = m;
        m_h1.resize(n);
        m_h1.setZero();
        m_h2.resize(m);
        m_h2.setZero();
        m_sigma.resize(n);
        for(auto & e: m_sigma)
        {
            e.clear();
            e.reserve(int(0.1 * m));
        }
        m_sigmat.resize(m);
        for(auto & e: m_sigmat)
        {
            e.clear();
            e.reserve(int(0.1 * n));
        }
        m_cache_sigmax.resize(n);
        m_cache_sigmatx.resize(m);
    }

    // Input gamma and M, compute the Hessian representations
    //
    // Also compute ||D||_F^2, D row sum, D column sum
    // D[i, j] = (alpha[i] + beta[j] - M[i, j])+, gamma = [alpha, beta]
    // Assume the memories of D_rowsum and D_colsum have been properly allocated
    void compute_D_hess(
        const Vector& gamma, const Matrix& M,
        double& D_sqnorm, double* D_rowsum, double* D_colsum
    );

    // res = sigma * x, x [m], res [n]
    void apply_sigmax(const double* x, double* res) const;

    // res = sigma' * x, x [n], res [m]
    void apply_sigmatx(const double* x, double* res) const;

    // Compute Delta * x
    // Delta * x = h2 .* x + lam * x - sigma' * ((sigma * x) ./ (h1 .+ lam))
    void apply_Deltax(const Vector& x, double shift, Vector& res) const;

    // Compute (H + shift * I) * x
    void apply_Hx(const Vector& x, double shift, Vector& res) const;
};


#endif  // REGOT_QROT_HESS_H
