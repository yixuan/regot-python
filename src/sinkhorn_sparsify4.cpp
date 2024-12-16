#include <iostream>
#include <numeric>  // std::accumulate
#include <chrono>
#include "sinkhorn_sparsify.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;

using Scalar = double;
using Index = int;

// Sparsify with given density
SpMat sparsify_mat4(
    const Matrix& T, double density
)
{
    // Step 1: Flatten the matrix into a vector
    std::vector<double> elements(T.data(), T.data() + T.size());

    // Step 2: Sort the elements
    std::sort(elements.begin(), elements.end());

    // Step 3: Determine the threshold for the top (100 * density)%
    size_t threshold_index = static_cast<size_t>((1 - density) * elements.size());
    double threshold = elements[threshold_index];

    // Step 4: Set elements below the threshold to 0
    return (T.array() > threshold).select(T, 0.0).sparseView();
}

// Sparsify with given density
Matrix sparsify_mat4_dense(
    const Matrix& T, double density
)
{
    // Step 1: Flatten the matrix into a vector
    std::vector<double> elements(T.data(), T.data() + T.size());

    // Step 2: Sort the elements
    std::sort(elements.begin(), elements.end());

    // Step 3: Determine the threshold for the top (100 * density)%
    size_t threshold_index = static_cast<size_t>((1 - density) * elements.size());
    double threshold = elements[threshold_index];

    // Step 4: Set elements below the threshold to 0
    return (T.array() > threshold).select(T, 0.0);
}


}  // namespace Sinkhorn
