#ifndef REGOT_QROT_CG_H
#define REGOT_QROT_CG_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "qrot_hess.h"

//============================= Wrapper =============================//
//
// https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
class HessianCG;

namespace Eigen {
namespace internal {
    // Inherit SparseMatrix's traints
    template <>
    struct traits<HessianCG>: public Eigen::internal::traits<Eigen::SparseMatrix<double>>
    {};
}
}

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

namespace Eigen {
namespace internal {
    template <typename Rhs>
    struct generic_product_impl<HessianCG, Rhs, SparseShape, DenseShape, GemvProduct>: // GEMV stands for matrix-vector
        generic_product_impl_base<HessianCG, Rhs, generic_product_impl<HessianCG, Rhs>>
    {
        using Scalar = typename Product<HessianCG, Rhs>::Scalar;

        template <typename Dest>
        static void scaleAndAddTo(Dest& dst, const HessianCG& lhs, const Rhs& rhs, const Scalar& alpha)
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



//========================= Preconditioner ==========================//
class HessianPrecond
{
private:
    using Vector = Eigen::VectorXd;

    Vector m_invdiag;
    mutable Vector m_cache;

public:
    HessianPrecond() {}

    template <typename MatrixType>
    explicit HessianPrecond(const MatrixType& mat)
    {
        compute(mat);
    }

    template <typename MatrixType>
    HessianPrecond& analyzePattern(const MatrixType&) { return *this; }

    // Typically MatrixType == HessianCG
    template <typename MatrixType>
    HessianPrecond& factorize(const MatrixType& mat)
    {
        m_invdiag.resize(mat.cols());
        m_invdiag.array() = 1.0 / (mat.hess().h2().array() + mat.shift());
        return *this;
    }

    template <typename MatrixType>
    HessianPrecond& compute(const MatrixType& mat) { return factorize(mat); }

    template <typename Rhs>
    const Vector& solve(const Rhs& b) const
    {
        m_cache.resizeLike(b);
        m_cache.array() = m_invdiag.array() * b.array();
        return m_cache;
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }
};
//========================= Preconditioner ==========================//



// (H + lam * I)^{-1} * rhs
void hess_cg(
    const Hessian& hess, const Eigen::VectorXd& rhs, double shift,
    Eigen::VectorXd& res, double tol = 1e-6,
    bool verbose = false, std::ostream& cout = std::cout
);


#endif  // REGOT_QROT_CG_H
