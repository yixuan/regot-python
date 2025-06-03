#include <iostream>
#include <vector>
#include <numeric>  // std::iota
#include "timer.h"
#include "sinkhorn_sparsify.h"

// Whether to print detailed timing information
// #define TIMING 1

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;
using Tri = Eigen::Triplet<double>;

using Scalar = double;
using Index = int;
using IndVec = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
using IndMat = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic>;

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

    std::nth_element(
        I, I + k, I + n,
        [x](Index i, Index j) {
            return x[i] < x[j];
        });
}

// Given data vector x(m) and index vector I(n), overwrite I such that
//     x_{I_0} <= ... <= x_{I_{n-1}}
inline void arg_sort(const Scalar* x, Index* I, Index n)
{
    // Early exit if n <= 0
    if (n <= 0)
        return;

    std::sort(
        I, I + n,
        [x](Index i, Index j) {
            return x[i] < x[j];
        }
    );
}

// 1. First do arg_select()
// 2. If x_{I_0} + ... + x_{I_{k-1}} <= target, return k
// 3. Otherwise, sort I_0, ..., I_{k-1} such that
//        x_{I_0} <= ... <= x_{I_{k-1}}
// 4. Find J such that
//        x_{I_0} + ... + x_{I_{J-1}} <= target < x_{I_0} + ... + x_{I_J}
inline Index arg_select_and_find(
    const Scalar* x, Index* I, Index n, Index k, Scalar target,
    Scalar& block_sum
)
{
    // Step 1
    arg_select(x, I, n, k);
    // Step 2
    block_sum = Scalar(0);
    for (Index i = 0; i < k; i++)
        block_sum += x[I[i]];
    if (block_sum <= target)
        return k;
    // Step 3
    arg_sort(x, I, k);
    // Step 4
    Scalar sum = Scalar(0);
    for (Index i = 0; i < k; i++)
    {
        sum += x[I[i]];
        if (sum > target)
            return i;
    }
    return k - 1;
}

// 1. First do arg_select()
// 2. If x_{I_k} + ... + x_{I_{n-1}} < target, return k-1
// 3. Otherwise, sort I_k, ..., I_{n-1} such that
//        x_{I_k} <= ... <= x_{I_{n-1}}
// 4. Find J such that
//        x_{I_{J+1}} + ... + x_{I_{n-1}} < target <= x_{I_J} + ... + x_{I_{n-1}}
inline Index arg_select_and_find_reverse(
    const Scalar* x, Index* I, Index n, Index k, Scalar target,
    Scalar& block_sum
)
{
    // Step 1
    arg_select(x, I, n, k);
    // Step 2
    block_sum = Scalar(0);
    for (Index i = n - 1; i >= k; i--)
        block_sum += x[I[i]];
    if (block_sum < target)
        return k - 1;
    // Step 3
    arg_sort(x, I + k, n - k);
    // Step 4
    Scalar sum = Scalar(0);
    for (Index i = n - 1; i >= k; i--)
    {
        sum += x[I[i]];
        if (sum >= target)
            return i;
    }
    return k;
}

// Given a data array x = [x_0, x_1, ..., x_{m-1}],
// an index set I = [I_0, ..., I_n], and a number delta,
// find an index subset J = [J_0, ..., J_{L-1}, J_L] such that
// {x_{J_0}, ..., x_{J_L}} are the smallest (L+1) elements of x,
// and x_{J_0} + ... + x_{J_{L-1}} <= delta < x_{J_0} + ... + x_{J_L}
//
// This function call returns L, and I will be overwritten as I'
// Elements of I are permutated such that J = [I'_0, ..., I'_L]
//
// If sum(x[I]) <= delta, then return L=n
//
inline Index find_small(
    const Scalar* x, Index* I, Index n, Scalar delta)
{
    // Block size
    Index m = Index(n / 32);
    m = std::max(m, Index(32));
    m = std::min(m, n);

    // Search from the left (smallest)
    // start-----working---working+bs-----end
    Index* start = I;
    Index* end = start + n;
    Index* working = start;
    Index* found = end;
    Scalar target = delta;
    for (; working < end; working += m)
    {
        Scalar block_sum;
        // The length of the array to do arg_select()
        Index len = std::distance(working, end);
        // Actual block size if <= m
        Index bs = std::min(len, m);
        Index L = arg_select_and_find(
            x, working, len, bs, target, block_sum);
        // If L < bs, then the index set has been found, given by
        // I -- (working + L - 1)
        if (L < bs)
        {
            found = working + L;
            break;
        }
        // Otherwise, it means we need to search in the next block
        // Adjust the target
        target -= block_sum;
    }

    return std::distance(start, found);
}

// Given a data array x = [x_0, x_1, ..., x_{m-1}],
// an index set I = [I_0, ..., I_n], and a number delta,
// find an index subset J = [J_0, ..., J_{L-1}, J_L] such that
// {x_{J_0}, ..., x_{J_L}} are the smallest (L+1) elements of x,
// and x_{J_0} + ... + x_{J_{L-1}} <= delta < x_{J_0} + ... + x_{J_L}
//
// This function call returns L, and I will be overwritten as I'
// Elements of I are permutated such that J = [I'_0, ..., I'_L]
//
// If sum(x[I]) <= delta, then return L=0
//
// We search from the right (largest), target = sum(x[I]) - delta
//
inline Index find_large(
    const Scalar* x, Index* I, Index n, Scalar target)
{
    // Block size
    Index m = Index(n / 32);
    m = std::max(m, Index(32));
    m = std::min(m, n);

    // Search from the right (largest)
    // start-----working-bs---working-----end
    Index* start = I;
    Index* end = start + n;
    Index* working = end;
    Index* found = start;
    for (; working >= start; working -= m)
    {
        Scalar block_sum;
        // The length of the array to do arg_select()
        Index len = std::distance(start, working);
        // Actual block size if <= m
        Index bs = std::min(len, m);
        Index k = std::distance(start, working - bs);
        Index L = arg_select_and_find_reverse(
            x, start, len, k, target, block_sum);
        // If L >= k, then the index set has been found, given by
        // I[0] -- I[L - 1]
        if (L >= k)
        {
            found = start + L;
            break;
        }
        // Otherwise, it means we need to search in the next block
        // Adjust the target
        target -= block_sum;
    }

    return std::distance(start, found);
}

// Sparsify a dense matrix, with the last column removed
SpMat sparsify_mat(const Matrix& T, double delta, double density_hint)
{
#ifdef TIMING
    Timer timer;
    timer.tic();
#endif

    // Dimensions
    const Index n = T.rows();
    const Index m = T.cols();

    // Index matrix for T
    // Each column of T_ind is an index vector for each column of T
    IndMat T_ind(n, m - 1);
    // First column
    std::iota(T_ind.data(), T_ind.data() + n, 0);
    // Copy to other columns
    for (Index j = 1; j < m - 1; j++)
    {
        Index offset = n * j;
        std::copy(T_ind.data(), T_ind.data() + n, T_ind.data() + offset);
    }

#ifdef TIMING
    timer.toc("T_ind");
#endif

    // Vector to store the partition index L
    IndVec L_col(m - 1);

    // Loop over columns
    #pragma omp parallel for schedule(static)
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        const Scalar* data = T.data() + offset;
        Index* I = T_ind.data() + offset;
        // Search from the smallest values
        if (density_hint > 0.1)
        {
            L_col[j] = find_small(data, I, n, delta);
        } else {
            // Search from the largest values
            Scalar sum = T.col(j).sum();
            Scalar target = sum - delta;
            L_col[j] = find_large(data, I, n, target);
        }
    }

#ifdef TIMING
    timer.toc("part_index");
#endif

    // Index vector of small values by row
    std::vector<std::vector<Index>> small_val_ind_row(n);
    for (Index i = 0; i < n; i++)
        small_val_ind_row[i].reserve(Index(0.1 * m));

    // Add small values to small_val_ind_row, and add large values to
    // sparse matrix triplet
    std::vector<Tri> tri_list;
    tri_list.reserve(Index(0.1 * n * m));
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        const Scalar* data = T.data() + offset;
        const Index* I = T_ind.data() + offset;
        // Small values
        Index L = L_col[j];
        for (Index l = 0; l < L; l++)
        {
            Index i = I[l];
            small_val_ind_row[i].emplace_back(j);
        }
        // Large values
        for (Index l = L; l < n; l++)
        {
            Index i = I[l];
            tri_list.emplace_back(i, j, data[i]);
        }
    }

    // No longer need T_ind, so free some memory
    T_ind.resize(1, 1);

#ifdef TIMING
    timer.toc("tri_list");
#endif

    // Find large elements in the small value vector
    // Control the rowwise error
    for (Index i = 0; i < n; i++)
    {
        Vector row = T.row(i).transpose();
        const Scalar* data = row.data();
        Index* I = small_val_ind_row[i].data();
        Index Ilen = small_val_ind_row[i].size();
        // Test row sum
        Scalar rowsum = 0.0;
        for (Index l = 0; l < Ilen; l++)
        {
            rowsum += data[I[l]];
        }
        // For simplicity of theoretical analysis we use > delta,
        // but it is OK to multiply delta with a constant to save computation
        if (rowsum > 2.0 * delta)
        {
            Scalar target = rowsum - 2.0 * delta;
            Index L = find_large(data, I, Ilen, target);

            // Add large values to the sparse matrix triplet
            for (Index l = L; l < Ilen; l++)
            {
                Index j = I[l];
                tri_list.emplace_back(i, j, data[j]);
            }
        }
    }

#ifdef TIMING
    timer.toc("row_sp");
#endif

    SpMat sp(n, m - 1);
    sp.setFromTriplets(tri_list.begin(), tri_list.end());

#ifdef TIMING
    timer.toc("to_sparse");
    std::cout << "T_ind = " << timer["T_ind"] <<
        ", part_index = " << timer["part_index"] <<
            ", tri_list = " << timer["tri_list"] << std::endl;
    std::cout << "row_sp = " << timer["row_sp"] <<
        ", to_sparse = " << timer["to_sparse"] << std::endl;
#endif

    // Matrix D = T.leftCols(m - 1) - sp;
    // std::cout << "delta = " << delta <<
    //     ", rowdelta = " << D.rowwise().sum().maxCoeff() <<
    //     ", coldelta = " << D.colwise().sum().maxCoeff() << std::endl;
    return sp;
}


}  // namespace Sinkhorn
