// #include <chrono>
#include "qrot_linsolve.h"
#include "qrot_cg.h"

namespace QROT {

using Vector = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
// // https://stackoverflow.com/a/34781413
// using Clock = std::chrono::high_resolution_clock;
// using Duration = std::chrono::duration<double, std::milli>;
// using TimePoint = std::chrono::time_point<Clock, Duration>;

// (H + lam * I + tau * K)^{-1} * rhs
// Method: 0 - CG
//         1 - SimplicialLDLT
//         2 - SimplicialLLT
//         3 - SparseLU

// A = H + lam * I, K = vv', v = (1n, 1m)
// (A + tau * vv')^(-1) * x = r - 1/(1/tau + v's) * A^(-1) * v * v' * r
// = r - (v'r) / (1/tau + v's) * A^(-1) * v = r - (v'r) / (1/tau + v's) * s
// r = A^(-1) * x, s = A^(-1) * v
template <typename Solver>
void direct_solver(
    Vector& res,
    Solver& solver, bool analyze_sparsity,
    const SpMat& Hs, double tau, const Vector& rhs,
    const int n, const int m
)
{
    // TimePoint clock_t1 = Clock::now();
    // This is typically the most time-consuming part
    if (analyze_sparsity)
        solver.analyzePattern(Hs);
    // TimePoint clock_t2 = Clock::now();
    solver.factorize(Hs);
    // TimePoint clock_t3 = Clock::now();
    res.noalias() = solver.solve(rhs);
    // TimePoint clock_t4 = Clock::now();
    // std::cout << "t2 - t1 = " << (clock_t2 - clock_t1).count() <<
    //     ", t3 - t2 = " << (clock_t3 - clock_t2).count() <<
    //     ", t4 - t3 = " << (clock_t4 - clock_t3).count() << std::endl;

    if (tau > 0.0)
    {
        Vector v(rhs.size()), s(rhs.size());
        v.head(n).setOnes();
        v.tail(m).setConstant(-1.0);
        s.noalias() = solver.solve(v);
        const double c1 = 1.0 / tau + v.dot(s);
        const double c2 = res.dot(v);
        res.array() -= (c2 / c1) * s.array();
    }
}

void QROTLinearSolver::solve(
    Vector& res,
    const Hessian& hess, const Vector& rhs, double shift,
    bool analyze_sparsity,
    std::ostream& cout
)
{
    if (this->method == 0)
    {
        // Call CG solver
        hess_cg(res, hess, rhs, shift, this->tau,
                this->cg_x0, this->cg_tol, this->verbose, cout);
    } else {
        // Construct sparse matrix
        const int n = hess.size_n();
        const int m = hess.size_m();
        SpMat I(n + m, n + m);
        I.setIdentity();
        SpMat Hs = hess.to_spmat() + shift * I;
        res.resizeLike(rhs);

        if (this->method == 2)
        {
            direct_solver(res, this->m_llt, analyze_sparsity,
                Hs, this->tau, rhs, n, m);
        } else if (this->method == 3) {
            direct_solver(res, this->m_lu, analyze_sparsity,
                Hs, this->tau, rhs, n, m);
        } else {
            direct_solver(res, this->m_ldlt, analyze_sparsity,
                Hs, this->tau, rhs, n, m);
        }
    }
}

}  // namespace QROT
