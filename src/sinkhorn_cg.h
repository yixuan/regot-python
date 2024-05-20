#ifndef REGOT_SINKHORN_CG_H
#define REGOT_SINKHORN_CG_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "sinkhorn_hess.h"

//============================= Wrapper =============================//
//
// https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
namespace Sinkhorn {

class HessianCG;

}  // namespace Sinkhorn


namespace Eigen {
namespace internal {
    // Inherit SparseMatrix's traints
    template <>
    struct traits<Sinkhorn::HessianCG>: public Eigen::internal::traits<Eigen::SparseMatrix<double>>
    {};
}
}


namespace Sinkhorn {

// Wrap Hessian to be used with Eigen::ConjugateGradient
class HessianCG: public Eigen::EigenBase<HessianCG>
{
private:
    using Index = Eigen::Index;

    const Hessian& m_hess;
    const double m_shift;

public:
    // Required fields
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = int;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    // Dimensions
    Index rows() const { return m_hess.size_m(); }
    Index cols() const { return m_hess.size_m(); }

    // Matrix product
    template <typename Rhs>
    Eigen::Product<HessianCG, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const
    {
        // Actual implementation below
        return Eigen::Product<HessianCG, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Constructor
    HessianCG(const Hessian& hess, double shift):
        m_hess(hess), m_shift(shift)
    {}

    // Const reference to members
    const Hessian& hess() const { return m_hess; }
    const double& shift() const { return m_shift; }
};

}  // namespace Sinkhorn


namespace Eigen {
namespace internal {
    template <typename Rhs>
    struct generic_product_impl<Sinkhorn::HessianCG, Rhs, SparseShape, DenseShape, GemvProduct>: // GEMV stands for matrix-vector
        generic_product_impl_base<Sinkhorn::HessianCG, Rhs, generic_product_impl<Sinkhorn::HessianCG, Rhs>>
    {
        using Scalar = typename Product<Sinkhorn::HessianCG, Rhs>::Scalar;

        template <typename Dest>
        static void scaleAndAddTo(Dest& dst, const Sinkhorn::HessianCG& lhs, const Rhs& rhs, const Scalar& alpha)
        {
            // This method should implement "dst += alpha * lhs * rhs" inplace,
            // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
            assert(alpha == Scalar(1) && "scaling is not implemented");
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);

            // Here we need to implement dst.noalias() += lhs * rhs,
            typename Dest::PlainObject res;
            lhs.hess().apply_Deltax(rhs, lhs.shift(), res);
            dst += res;
        }
    };
}
}
//============================= Wrapper =============================//



namespace Sinkhorn {

// (Hs + shift * I)^{-1} * rhs
void hess_cg(
    Eigen::VectorXd& res,
    const Hessian& hess, const Eigen::VectorXd& rhs, double shift,
    const Eigen::VectorXd& guess, double tol = 1e-8,
    bool verbose = false, std::ostream& cout = std::cout
);

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_CG_H
