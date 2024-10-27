#include "qrot_cg.h"
#include <Eigen/IterativeLinearSolvers>

namespace QROT {

using Vector = Eigen::VectorXd;

// (H + lam * I)^{-1} * rhs
void hess_cg(
    Vector& res,
    const Hessian& hess, const Vector& rhs, double shift, double tau,
    const Vector& guess, double tol,
    int verbose, std::ostream& cout
)
{
    const int n = hess.size_n();
    const int m = hess.size_m();
    res.resizeLike(rhs);

    // x = [w, z], w [n], z [m]
    // v = A^(-1) * w
    Vector v(n);
    hess.solve_Ax(rhs.head(n), shift, tau, v);
    // y = C * v
    Vector y(m);
    hess.apply_Cx(v, tau, y);
    // r2 = Delta^{-1} * (z - y)
    Vector zy(m);
    zy.noalias() = rhs.tail(m) - y;
    HessianCG hesscg(hess, shift, tau);
    Eigen::ConjugateGradient<HessianCG, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    // Eigen::ConjugateGradient<HessianCG, Eigen::Lower | Eigen::Upper, HessianPrecond> cg;
    cg.compute(hesscg);
    cg.setTolerance(tol);
    if (guess.size() == m)
        res.tail(m).noalias() = cg.solveWithGuess(zy, guess);
    else
        res.tail(m).noalias() = cg.solve(zy);
    if (verbose >= 2)
    {
        cout << "[cg]=====================================================" << std::endl;
        cout << "║ info = " << cg.info() << ", niter = " << cg.iterations() << std::endl;
        cout << "=========================================================" << std::endl << std::endl;
    }
    // r1 = A^(-1) * (w - B * r2)
    // y <- B * r2, r1 <- w - y, y <- A^(-1) * r1, r1 <- y
    hess.apply_Bx(res.tail(m), tau, y);
    res.head(n).noalias() = rhs.head(n) - y;
    hess.solve_Ax(res.head(n), shift, tau, y);
    res.head(n).noalias() = y;
}

}  // namespace QROT
