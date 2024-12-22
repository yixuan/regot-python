#ifndef REGOT_SINKHORN_LINSOLVE_H
#define REGOT_SINKHORN_LINSOLVE_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/LU> // inverse
#include <Eigen/SparseCholesky>
#include "sinkhorn_hess.h"

namespace Sinkhorn {

// (Hs + lam * I)^{-1} * rhs
// Method: 0 - CG
//         1 - SimplicialLDLT
//         2 - SimplicialLLT
//         3 - SparseLU
struct SinkhornLinearSolver
{
private:
    using Vector = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<double>;

    // Linear solvers
    Eigen::SimplicialLDLT<SpMat> m_ldlt;
    Eigen::SimplicialLLT<SpMat> m_llt;
    Eigen::SparseLU<SpMat> m_lu;

public:
    // Parameters
    int method;
    Vector cg_x0;
    double cg_tol;
    bool verbose;

    // Setting default values
    SinkhornLinearSolver():
        method(0), cg_x0(0), cg_tol(1e-8), verbose(false)
    {}

    void solve(
        Vector& res,
        const Hessian& hess, const Vector& rhs, double shift,
        bool analyze_sparsity = true,
        std::ostream& cout = std::cout
    );

    void solve_low_rank(
        Vector& res,
        const Hessian& hess, const Vector& rhs,
        double shift, const Vector& y, const Vector& s,
        std::ostream& cout = std::cout
    );
};

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_LINSOLVE_H
