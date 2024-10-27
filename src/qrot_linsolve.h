#ifndef REGOT_QROT_LINSOLVE_H
#define REGOT_QROT_LINSOLVE_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include "qrot_hess.h"

namespace QROT {

// (H + lam * I)^{-1} * rhs
// Method: 0 - CG
//         1 - SimplicialLDLT
//         2 - SimplicialLLT
//         3 - SparseLU
struct QROTLinearSolver
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
    double tau;
    Vector cg_x0;
    double cg_tol;
    int verbose;

    // Setting default values
    QROTLinearSolver():
        method(0), tau(0.0), cg_x0(0), cg_tol(1e-8), verbose(0)
    {}

    void solve(
        Vector& res,
        const Hessian& hess, const Vector& rhs, double shift,
        bool analyze_sparsity = true,
        std::ostream& cout = std::cout
    );
};

}  // namespace QROT


#endif  // REGOT_QROT_LINSOLVE_H
