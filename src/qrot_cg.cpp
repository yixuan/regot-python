#include "qrot_cg.h"
#include <Eigen/IterativeLinearSolvers>

using Vector = Eigen::VectorXd;

// (H + lam * I)^{-1} * rhs
void hess_cg(
    const Hessian& hess, const Vector& rhs, double shift,
    Vector& res, double tol, bool verbose, std::ostream& cout
)
{
    const int n = hess.size_n();
    const int m = hess.size_m();
    res.resizeLike(rhs);

    // x = [w, z], w [n], z [m]
    // v = w ./ (h1 .+ lam)
    Vector v(n);
    v.array() = rhs.head(n).array() / (hess.h1().array() + shift);
    // y = sigma' * v, use the memory of res.tail(m)
    hess.apply_sigmatx(v.data(), res.data() + n);
    // r2 = Delta^{-1} * (z - y)
    Vector zy(m);
    zy.noalias() = rhs.tail(m) - res.tail(m);
    HessianCG hesscg(hess, shift);
    // Eigen::ConjugateGradient<HessianCG, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    Eigen::ConjugateGradient<HessianCG, Eigen::Lower | Eigen::Upper, HessianPrecond> cg;
    cg.compute(hesscg);
    cg.setTolerance(tol);
    res.tail(m).noalias() = cg.solve(zy);
    if(verbose)
    {
        cout << "CG info = " << cg.info() <<
            ", CG niter = " << cg.iterations() << std::endl;
    }
    // sigma * r2, use the memory of res.head(n)
    hess.apply_sigmax(res.data() + n, res.data());
    // r1 = v - (sigma * r2) ./ (h1 .+ lam)
    res.head(n).array() = v.array() - res.head(n).array() / (hess.h1().array() + shift);
}
