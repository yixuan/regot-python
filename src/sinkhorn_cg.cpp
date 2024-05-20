#include "sinkhorn_cg.h"
#include <Eigen/IterativeLinearSolvers>

namespace Sinkhorn {

using Vector = Eigen::VectorXd;

// (H + lam * I)^{-1} * rhs
void hess_cg(
    Vector& res,
    const Hessian& hess, const Vector& rhs, double shift,
    const Vector& guess, double tol,
    bool verbose, std::ostream& cout
)
{
    const int n = hess.size_n();
    const int m = hess.size_m();
    res.resizeLike(rhs);

    // x = [w, z], w [n], z [m-1]
    // v = A^(-1) * w
    Vector v(n);
    hess.solve_Ax(rhs.head(n), shift, v);
    // y = C * v
    Vector y(m - 1);
    hess.apply_Cx(v, y);
    // r2 = Delta^{-1} * (z - y)
    Vector zy(m - 1);
    zy.noalias() = rhs.tail(m - 1) - y;
    HessianCG hesscg(hess, shift);
    Eigen::ConjugateGradient<HessianCG, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    // Eigen::ConjugateGradient<HessianCG, Eigen::Lower | Eigen::Upper, HessianPrecond> cg;
    cg.compute(hesscg);
    cg.setTolerance(tol);
    if (guess.size() == m - 1)
        res.tail(m - 1).noalias() = cg.solveWithGuess(zy, guess);
    else
        res.tail(m - 1).noalias() = cg.solve(zy);
    if(verbose)
    {
        cout << "CG info = " << cg.info() <<
            ", CG niter = " << cg.iterations() << std::endl;
    }
    // r1 = A^(-1) * (w - B * r2)
    // y <- B * r2, r1 <- w - y, y <- A^(-1) * r1, r1 <- y
    hess.apply_Bx(res.tail(m - 1), y);
    res.head(n).noalias() = rhs.head(n) - y;
    hess.solve_Ax(res.head(n), shift, y);
    res.head(n).noalias() = y;
}

}  // namespace Sinkhorn
