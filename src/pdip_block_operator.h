// Matrix-free block Schur operator for PDIP-CG inner solves (Eigen::ConjugateGradient).
// Pattern: https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
// (same idea as qrot_cg.h HessianCG / sinkhorn_cg.h)

#ifndef REGOT_PDIP_BLOCK_OPERATOR_H
#define REGOT_PDIP_BLOCK_OPERATOR_H

#include <cassert>
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace PDIP {

class BlockACG;

}  // namespace PDIP

namespace Eigen {
namespace internal {
template <>
struct traits<PDIP::BlockACG> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}  // namespace internal
}  // namespace Eigen

namespace PDIP {

using Vector = Eigen::VectorXd;

// A maps [w; z] with w in R^M, z in R^{n-1}:
//   y1 = diag(B11)*w + D*z,  y2 = diag(B22)*z + D'*w,  D = reshape(D_B12, M, n12) column-major.
class BlockACG : public Eigen::EigenBase<BlockACG> {
private:
    using Index = Eigen::Index;

    int M_dem_;
    int n12_;
    Eigen::Ref<const Vector> B11_;
    Eigen::Ref<const Vector> B22_;
    Eigen::Ref<const Vector> D_B12_;

public:
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = int;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return M_dem_ + n12_; }
    Index cols() const { return M_dem_ + n12_; }

    int M_dem() const { return M_dem_; }
    int n12() const { return n12_; }
    const Eigen::Ref<const Vector>& B11() const { return B11_; }
    const Eigen::Ref<const Vector>& B22() const { return B22_; }
    const Eigen::Ref<const Vector>& D_B12() const { return D_B12_; }

    template <typename Rhs>
    Eigen::Product<BlockACG, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<BlockACG, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    BlockACG(int M_dem, int n12, const Vector& B11, const Vector& B22, const Vector& D_B12)
        : M_dem_(M_dem), n12_(n12), B11_(B11), B22_(B22), D_B12_(D_B12) {}
};

}  // namespace PDIP

namespace Eigen {
namespace internal {

template <typename Rhs>
struct generic_product_impl<PDIP::BlockACG, Rhs, SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<PDIP::BlockACG, Rhs, generic_product_impl<PDIP::BlockACG, Rhs>> {
    using Scalar = typename Product<PDIP::BlockACG, Rhs>::Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const PDIP::BlockACG& lhs, const Rhs& rhs, const Scalar& alpha) {
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        const int M = lhs.M_dem();
        const int n12 = lhs.n12();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> xvec = rhs;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> y(M + n12);
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> D(
            lhs.D_B12().data(), M, n12);
        y.head(M).noalias() = lhs.B11().cwiseProduct(xvec.head(M)) + D * xvec.tail(n12);
        y.tail(n12).noalias() = lhs.B22().cwiseProduct(xvec.tail(n12)) + D.transpose() * xvec.head(M);
        dst += y;
    }
};

}  // namespace internal
}  // namespace Eigen

#endif  // REGOT_PDIP_BLOCK_OPERATOR_H
