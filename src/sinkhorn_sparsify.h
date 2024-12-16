#ifndef REGOT_SINKHORN_SPARSIFY_H
#define REGOT_SINKHORN_SPARSIFY_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;

// Sparsify a dense matrix, with the last column removed
SpMat sparsify_mat(
    const Matrix& T, double delta, double density_hint
);

SpMat sparsify_mat2(
    const Matrix& T, double delta, double density_hint
);

SpMat sparsify_mat3(
    const Matrix& T, double delta, double density_hint
);

// Sparsify with given density
SpMat sparsify_mat4(
    const Matrix& T, double density
);

// Sparsify with given density
Matrix sparsify_mat4_dense(
    const Matrix& T, double density
);

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_SPARSIFY_H
