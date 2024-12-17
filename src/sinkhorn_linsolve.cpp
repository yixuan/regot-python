#include <chrono>
#include "sinkhorn_linsolve.h"
#include "sinkhorn_cg.h"
// #include <unsupported/Eigen/SparseExtra>

// Whether to print detailed timing information
// #define TIMING 1

namespace Sinkhorn {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

#ifdef TIMING
// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;
#endif

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
    TimePoint clock_t1 = Clock::now();
#endif

    // This is typically one of the most time-consuming parts
    if (analyze_sparsity)
        solver.analyzePattern(Hl);

#ifdef TIMING
    TimePoint clock_t2 = Clock::now();
#endif

    solver.factorize(Hl);

#ifdef TIMING
    TimePoint clock_t3 = Clock::now();
#endif

    res.noalias() = solver.solve(rhs);

#ifdef TIMING
    TimePoint clock_t4 = Clock::now();
    std::cout << "analyze = " << (clock_t2 - clock_t1).count() <<
        ", factorize = " << (clock_t3 - clock_t2).count() <<
        ", solve = " << (clock_t4 - clock_t3).count() << std::endl;
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
        TimePoint clock_t1 = Clock::now();
#endif

        SpMat Hl = hess.to_spmat(only_lower) + shift * I;

#ifdef TIMING
        TimePoint clock_t2 = Clock::now();
        std::cout << "[linsolve]================================================" << std::endl;
        std::cout << "to_sparse = " << (clock_t2 - clock_t1).count() << std::endl;
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

void SinkhornLinearSolver::solve_sr2(
    Vector& res,
    const Hessian& hess, const Vector& rhs,
    double shift, const Vector& y, const Vector& s,
    bool analyze_sparsity,
    std::ostream& cout
)
{

    /* Stupid Dense implementation */
    // const int n = hess.size_n();
    // const int m = hess.size_m();
    // SpMat I(n + m - 1, n + m - 1);
    // I.setIdentity();
    // Matrix Hl = (hess.to_spmat(false) + shift * I).toDense();
    // Vector u = y, v = Hl * s;
    // double a = 1.0 / u.dot(s), b = -1.0 / v.dot(s);
    // res = (Hl + (a * u) * u.transpose() + (b * v) * v.transpose()).llt().solve(rhs);


    /* Stupid Sparse implementation */
    // const int n = hess.size_n();
    // const int m = hess.size_m();
    // SpMat I(n + m - 1, n + m - 1);
    // I.setIdentity();
    // SpMat Hl = hess.to_spmat(false) + shift * I;
    // Vector u = y, v = Hl * s;
    // double a = 1.0 / u.dot(s), b = -1.0 / v.dot(s);
    // Eigen::SimplicialLLT<SpMat> solver;
    // solver.compute(
    //     Hl + ((a * u) * u.transpose()).sparseView()
    //        + ((b * v) * v.transpose()).sparseView()
    // );
    // res = solver.solve(rhs);


    /* Improved Sparse implementation */
    // Construct sparse matrix
    const int n = hess.size_n();
    const int m = hess.size_m();
    SpMat I(n + m - 1, n + m - 1);
    I.setIdentity();

    // Hl = Hs + shift * I
    SpMat Hl = hess.to_spmat(false) + shift * I;

    // /* intermediate variables */
    Vector u = y, v = Hl * s;
    /*
    O = [
        [u' * Hl^{-1} * u + 1/a, u' * Hl^{-1} * v],
        [v' * Hl^{-1} * u, v' * Hl^{-1} * v + 1/b]
    ]^{-1}
    */
    Eigen::Matrix2d O;
    /*
    P = [Hl^{-1} * u, Hl^{-1} * v]
    Q = [u, v]
    */
    Eigen::MatrixX2d P, Q;
    /* Define solver */
    Eigen::SimplicialLLT<SpMat> solver;
    solver.compute(Hl);

    /* Solve intermediate values */
    Vector Hl_inv_rhs = solver.solve(rhs);
    Vector Hl_inv_u = solver.solve(u);
    Vector Hl_inv_v = solver.solve(v);

    // Compute O
    O(0, 0) = u.dot(Hl_inv_u) + u.dot(s);
    O(0, 1) = u.dot(Hl_inv_v);
    O(1, 0) = O(0, 1);
    O(2, 2) = v.dot(Hl_inv_v) + v.dot(s);
    O = O.inverse();

    // P = [Hl^{-1} * u, Hl^{-1} * v]
    P.col(0) = Hl_inv_u;
    P.col(1) = Hl_inv_v;

    // Q = [u, v]
    Q.col(0) = u;
    Q.col(1) = v;
    
    res = Hl_inv_rhs - P * O * Q.transpose() * Hl_inv_rhs;
}

}  // namespace Sinkhorn

