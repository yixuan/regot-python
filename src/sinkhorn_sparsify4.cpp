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

// Sparsify with given density, and remove the last column
SpMat sparsify_mat4(
    const Matrix& T, double density
)
{
    // cpp argsort: https://gist.github.com/HViktorTsoi/58eabb4f7c5a303ced400bcfa816f6f5
    const int n = T.rows();
    const int m = T.cols();

    // remove the last column and flatten the matrix into a vector
    Matrix Tt = T.leftCols(m - 1);
    std::vector<double> elements(Tt.data(), Tt.data() + Tt.size());

    // find the threshold
    int num_zeros = Tt.size() * (1 - density);
    std::nth_element(elements.begin(), elements.begin() + num_zeros, elements.end());
    double threshold = elements[num_zeros];
    
    return Tt.sparseView(threshold, 1.0);
}

}  // namespace Sinkhorn
