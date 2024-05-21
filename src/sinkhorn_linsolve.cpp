// #include <chrono>
#include "sinkhorn_linsolve.h"
#include "sinkhorn_cg.h"

namespace Sinkhorn {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
// // https://stackoverflow.com/a/34781413
// using Clock = std::chrono::high_resolution_clock;
// using Duration = std::chrono::duration<double, std::milli>;
// using TimePoint = std::chrono::time_point<Clock, Duration>;

// (Hs + lam * I)^{-1} * rhs, Hl = Hs + lam * I
// Method: 0 - CG
//         1 - SimplicialLDLT
//         2 - SimplicialLLT
//         3 - SparseLU
template <typename Solver>
void direct_solver(
    Vector& res,
    Solver& solver, bool analyze_sparsity,
    const SpMat& Hl, const Vector& rhs,
    const int n, const int m
)
{
    // TimePoint clock_t1 = Clock::now();
    // This is typically the most time-consuming part
    if (analyze_sparsity)
        solver.analyzePattern(Hl);
    // TimePoint clock_t2 = Clock::now();
    solver.factorize(Hl);
    // TimePoint clock_t3 = Clock::now();
    res.noalias() = solver.solve(rhs);
    // TimePoint clock_t4 = Clock::now();
    // std::cout << "t2 - t1 = " << (clock_t2 - clock_t1).count() <<
    //     ", t3 - t2 = " << (clock_t3 - clock_t2).count() <<
    //     ", t4 - t3 = " << (clock_t4 - clock_t3).count() << std::endl;
}

void SinkhornLinearSolver::solve(
    Vector& res,
    const Hessian& hess, const Vector& rhs, double shift,
    bool analyze_sparsity,
    std::ostream& cout
)
{
    if (this->method == 0)
    {
        // Call CG solver
        hess_cg(res, hess, rhs, shift,
                this->cg_x0, this->cg_tol, this->verbose, cout);
    } else {
        // Construct sparse matrix
        const int n = hess.size_n();
        const int m = hess.size_m();
        SpMat I(n + m - 1, n + m - 1);
        I.setIdentity();
        bool only_lower = (this->method != 3);
        // TimePoint clock_t1 = Clock::now();
        SpMat Hl = hess.to_spmat(only_lower) + shift * I;
        // TimePoint clock_t2 = Clock::now();
        // std::cout << "tosparse = " << (clock_t2 - clock_t1).count() << std::endl;
        res.resizeLike(rhs);

        if (this->method == 2)
        {
            direct_solver(res, this->m_llt, analyze_sparsity,
                Hl, rhs, n, m);
        } else if (this->method == 3) {
            direct_solver(res, this->m_lu, analyze_sparsity,
                Hl, rhs, n, m);
        } else {
            direct_solver(res, this->m_ldlt, analyze_sparsity,
                Hl, rhs, n, m);
        }
    }
}

}  // namespace Sinkhorn
