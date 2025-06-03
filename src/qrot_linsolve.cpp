#include "timer.h"
#include "qrot_linsolve.h"
#include "qrot_cg.h"

// Whether to print detailed timing information
// #define TIMING 1

namespace QROT {

using Vector = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

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
#ifdef TIMING
    Timer timer;
    timer.tic();
#endif

    // This is typically the most time-consuming part
    if (analyze_sparsity)
        solver.analyzePattern(Hs);

#ifdef TIMING
    timer.toc("analyze");
#endif

    solver.factorize(Hs);

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
