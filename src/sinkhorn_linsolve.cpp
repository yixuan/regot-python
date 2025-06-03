#include "timer.h"
#include "sinkhorn_linsolve.h"
#include "sinkhorn_cg.h"
// #include <unsupported/Eigen/SparseExtra>

// Whether to print detailed timing information
// #define TIMING 1

namespace Sinkhorn {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

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
#ifdef TIMING
    Timer timer;
    timer.tic();
#endif

    // This is typically one of the most time-consuming parts
    if (analyze_sparsity)
        solver.analyzePattern(Hl);

#ifdef TIMING
    timer.toc("analyze");
#endif

    solver.factorize(Hl);

#ifdef TIMING
    timer.toc("factorize");
#endif

    res.noalias() = solver.solve(rhs);

#ifdef TIMING
    timer.toc("solve");
    std::cout << "analyze = " << timer["analyze"] <<
        ", factorize = " << timer["factorize"] <<
        ", solve = " << timer["solve"] << std::endl;
    std::cout << "==========================================================" << std::endl;
#endif
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

#ifdef TIMING
        Timer timer;
        timer.tic();
#endif

        SpMat Hl = hess.to_spmat(only_lower) + shift * I;

#ifdef TIMING
        double duration = timer.toc("to_sparse");
        std::cout << "[linsolve]================================================" << std::endl;
        std::cout << "to_sparse = " << duration << std::endl;
        // Eigen::saveMarket(Hl, "mat.mtx");
#endif

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

// Solve (H + UCV + shift * I)x = rhs
// A = H + shift * I
// U = [u, v], C = diag(a, b), V = U'
// u = y, v = A * s
// a = 1 / u's, b = -1 / v's
/*
void SinkhornLinearSolver::solve_low_rank(
    Vector& res,
    const Hessian& hess, const Vector& rhs,
    double shift, const Vector& y, const Vector& s,
    std::ostream& cout
)
{
    res.resizeLike(rhs);

    // Construct sparse matrix
    const int n = hess.size_n();
    const int m = hess.size_m();
    SpMat I(n + m - 1, n + m - 1);
    I.setIdentity();
    const bool only_lower = true;
    SpMat A = hess.to_spmat(only_lower) + shift * I;

    // Intermediate variables for Woodbury formula
    Vector u = y, v = A.selfadjointView<Eigen::Lower>() * s;
    const double inva = u.dot(s), invb = -v.dot(s);

    // U = [u, v], V = U'
    Matrix U(n + m - 1, 2);
    U.col(0).noalias() = u;
    U.col(1).noalias() = v;

    // A = H + shift * I
    // inv(A + UCV) = inv(A) - inv(A) * U * inv(middle) * V * inv(A)
    // middle = inv(C) + V * inv(A) * U

    // Construct LDLT Solver
    Eigen::SimplicialLDLT<SpMat> solver;
    solver.compute(A);

    // Solve the sparse linear system
    Vector invA_rhs = solver.solve(rhs);
    Matrix invA_U = solver.solve(U);
    Eigen::Matrix2d invC {{inva, 0.0}, {0.0, invb}};
    Eigen::Matrix2d middle = invC + U.transpose() * invA_U;

    res.noalias() = invA_rhs - invA_U * (middle.lu().solve(U.transpose() * invA_rhs));
}
*/

// BFGS update rule
// https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
void SinkhornLinearSolver::solve_low_rank(
    Vector& res,
    const Hessian& hess, const Vector& rhs,
    double shift, const Vector& y, const Vector& s,
    bool analyze_sparsity,
    std::ostream& cout
)
{
#ifdef TIMING
    Timer timer;
    timer.tic();
#endif

    res.resizeLike(rhs);

    // Construct sparse matrix
    const int n = hess.size_n();
    const int m = hess.size_m();
    SpMat I(n + m - 1, n + m - 1);
    I.setIdentity();
    const bool only_lower = true;
    SpMat A = hess.to_spmat(only_lower) + shift * I;

#ifdef TIMING
    timer.toc("to_sparse");
#endif

    // Construct LDLT Solver
    Eigen::SimplicialLDLT<SpMat> solver;
    // This is typically one of the most time-consuming parts
    if (analyze_sparsity)
        solver.analyzePattern(A);

#ifdef TIMING
    timer.toc("analyze");
#endif

    solver.factorize(A);

#ifdef TIMING
    timer.toc("factorize");
#endif

    // Solve the sparse linear system
    Vector invA_rhs = solver.solve(rhs);
    Vector invA_y = solver.solve(y);

    const double rs = s.dot(rhs), ys = y.dot(s),
        yinvAy = y.dot(invA_y), yinvAr = y.dot(invA_rhs);
    const double rs_ys = rs / ys;

    // BFGS rule
    res.noalias() = invA_rhs + ((1.0 + yinvAy / ys) * rs_ys - yinvAr / ys) * s -
        rs_ys * invA_y;

    // DFP rule
    // res.noalias() = invA_rhs - (yinvAr / yinvAy) * invA_y + rs_ys * s;

#ifdef TIMING
    timer.toc("solve");
    cout << "to_sparse = " << timer["to_sparse"] <<
        ", analyze = " << timer["analyze"] <<
        ", factorize = " << timer["factorize"] <<
        ", solve = " << timer["solve"] << std::endl;
#endif
}

}  // namespace Sinkhorn
