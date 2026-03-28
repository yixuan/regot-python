// PDIP transport operators: A and A^T for the OT equality stack, and solve for the
// structured A A^T system. Layout matches CG/FP: x is row-major n×m flattened (index i*m+j).
#ifndef REGOT_PDIP_TRANSPORT_OPS_H
#define REGOT_PDIP_TRANSPORT_OPS_H

#include <Eigen/Core>

namespace PDIP {
namespace transport {

using MatRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// y <- A x,  y in R^{m+n-1}, x stacked as n×m row-major
inline void A_matvec_from_x(int n, int m, const double* x, const double* scale_or_null, double* y) {
    Eigen::Map<const MatRowMajor> X(x, n, m);
    Eigen::Map<Eigen::VectorXd> yv(y, m + n - 1);
    if (scale_or_null) {
        Eigen::Map<const MatRowMajor> S(scale_or_null, n, m);
        MatRowMajor XS = X.cwiseProduct(S);
        yv.head(m) = XS.colwise().sum().transpose();
        yv.tail(n - 1) = XS.topRows(n - 1).rowwise().sum();
    } else {
        yv.head(m) = X.colwise().sum().transpose();
        yv.tail(n - 1) = X.topRows(n - 1).rowwise().sum();
    }
}

// y <- A^T lambda,  y in R^{n*m}, lambda in R^{m+n-1}
inline void AT_matvec(int n, int m, const double* lambda, double* y) {
    Eigen::Map<const Eigen::VectorXd> lam(lambda, m + n - 1);
    Eigen::Map<MatRowMajor> Y(y, n, m);
    const Eigen::RowVectorXd row_base = lam.head(m).transpose();
    for (int i = 0; i < n; ++i) {
        const double lam_i = (i < n - 1 ? lam(m + i) : 0.0);
        Y.row(i) = row_base.array() + lam_i;
    }
}

// Structured solve used for initial lambda and similar; rhs length m+n-1, out same.
inline void solve_AAT(const double* rhs, int n, int m, double* out) {
    Eigen::Map<const Eigen::VectorXd> r(rhs, m + n - 1);
    Eigen::Map<Eigen::VectorXd> o(out, m + n - 1);
    const double sum_d = r.head(m).sum();
    Eigen::VectorXd tmp = Eigen::VectorXd::Constant(n - 1, sum_d / static_cast<double>(n)) - r.tail(n - 1);
    const double sum_t = tmp.sum();
    o.tail(n - 1) = (tmp.array() + sum_t) / static_cast<double>(m);
    const double sum_x2 = o.tail(n - 1).sum();
    o.head(m) = (r.head(m).array() - sum_x2) / static_cast<double>(n);
}

inline void solve_AAT_vec(const Eigen::Ref<const Eigen::VectorXd>& rhs_in, int n_supp, int m_dem, Eigen::Ref<Eigen::VectorXd> out) {
    const double sum_d = rhs_in.head(m_dem).sum();
    Eigen::VectorXd tmp = Eigen::VectorXd::Constant(n_supp - 1, sum_d / static_cast<double>(n_supp)) - rhs_in.segment(m_dem, n_supp - 1);
    const double sum_t = tmp.sum();
    out.tail(n_supp - 1) = (tmp.array() + sum_t) / static_cast<double>(m_dem);
    const double sum_x2 = out.tail(n_supp - 1).sum();
    out.head(m_dem) = (rhs_in.head(m_dem).array() - sum_x2) / static_cast<double>(n_supp);
}

}  // namespace transport
}  // namespace PDIP

#endif
