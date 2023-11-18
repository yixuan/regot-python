#ifndef REGOT_QROT_APPROX_PROJ_H
#define REGOT_QROT_APPROX_PROJ_H

#include <Eigen/Core>

template <typename ResultMatrix>
ResultMatrix approx_proj(
    const ResultMatrix& plan,
    const Eigen::Ref<const Eigen::VectorXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b
)
{
    using Index = Eigen::Index;
    using Vector = Eigen::VectorXd;

    const Index n = plan.rows();
    const Index m = plan.cols();
    Vector x = a.cwiseQuotient(plan.rowwise().sum()).cwiseMin(1.0);
    ResultMatrix res = plan.cwiseProduct(x.replicate(1, m));
    Vector y = b.cwiseQuotient(res.colwise().sum().transpose()).cwiseMin(1.0);
    res.noalias() = res.cwiseProduct(y.transpose().replicate(n, 1));
    Vector err_a = a - res.rowwise().sum();
    Vector err_b = b - res.colwise().sum().transpose();
    res.noalias() += err_a * err_b.transpose() / err_a.cwiseAbs().sum();
    return res;
}


#endif  // REGOT_QROT_APPROX_PROJ_H
