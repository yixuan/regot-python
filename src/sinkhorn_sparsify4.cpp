#include <iostream>
#include <vector>
#include <numeric>  // std::accumulate, std::iota
#include <chrono>
#include "sinkhorn_sparsify.h"

// Fast sorting functions
#ifdef __AVX2__
#include "sorting/xss-common-argsort.h"
#include "sorting/avx2-64bit-qsort.hpp"
#endif

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;
using Tri = Eigen::Triplet<double>;

using Scalar = double;
using Index = std::size_t;
using IndVec = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

// Sparsify with given density, and remove the last column
SpMat sparsify_mat4_old(
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

// Given data vector x(m) and index vector I(n), overwrite I such that
// I is partitioned into two parts:
//     I_0, ..., I_{k-1} | I_k, ..., I_{n-1}
// We require that for any i in {I_0, ..., I_{k-1}} and j in {I_k, ..., I_{n-1}},
//     x_i <= x_j
inline void arg_select(const Scalar* x, Index* I, Index n, Index k)
{
    // Early exit if n <= 0 or k <= 0 or k >= n
    if (n <= 0 || k <= 0 || k >= n)
        return;

#ifdef __AVX2__
    // avx2_argselect() needs a non-const data pointer
    avx2_argselect(const_cast<Scalar*>(x), I, k, n);
#else
    std::nth_element(
        I, I + k, I + n,
        [x](Index i, Index j) {
            return x[i] < x[j];
        });
#endif
}

SpMat sparsify_mat4(
    const Matrix& T, double density
)
{
    // Dimensions
    const Index n = T.rows();
    const Index m = T.cols();

    // Constuct index vector
    // We only consider the first (m-1) columns of T
    Index ind_len = n * (m - 1);
    IndVec ind(ind_len);
    std::iota(ind.data(), ind.data() + ind_len, 0);

    // Select the largest k elements
    density = std::min(density, 1.0);
    density = std::max(density, 0.0);
    Index k = Index(ind_len * density);

    // Note that arg_select() partially sorts data in increasing order
    // So we need to put (ind_len - k) elements to the left
    arg_select(T.data(), ind.data(), ind_len, ind_len - k);

    // Now the last k elements in ind contains the indices of the
    // largest k elements in T[:, :-1]
    // We can recover the row/column indices as i % n and i // n, respectively
    std::vector<Tri> tri_list;
    tri_list.reserve(k + 1);
    const Index* ind_start = ind.data() + (ind_len - k);
    const Index* ind_end = ind_start + k;
    const Scalar* data_start = T.data();
    const Index* ind_ptr = ind_start;
    for (; ind_ptr < ind_end; ind_ptr++)
    {
        const Index i = (*ind_ptr);
        const Index row = i % n;
        const Index col = i / n;
        const Scalar val = data_start[i];
        tri_list.emplace_back(row, col, val);
    }

    SpMat sp(n, m - 1);
    sp.setFromTriplets(tri_list.begin(), tri_list.end());
    sp.makeCompressed();

    return sp;
}

}  // namespace Sinkhorn
