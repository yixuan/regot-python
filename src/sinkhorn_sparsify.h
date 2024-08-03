#ifndef REGOT_SINKHORN_SPARSIFY_H
#define REGOT_SINKHORN_SPARSIFY_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace Sinkhorn {

// Sparsify a dense matrix, with the last column removed
Eigen::SparseMatrix<double> sparsify_mat(
    const Eigen::MatrixXd& T, double delta, double density_hint
);

Eigen::SparseMatrix<double> sparsify_mat2(
    const Eigen::MatrixXd& T, double delta, double density_hint
);

Eigen::SparseMatrix<double> sparsify_mat3(
    const Eigen::MatrixXd& T, double delta, double density_hint
);

}  // namespace Sinkhorn


#endif  // REGOT_SINKHORN_SPARSIFY_H
