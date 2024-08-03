#include <iostream>
#include <vector>
#include <chrono>
#include "sinkhorn_sparsify.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;
using Tri = Eigen::Triplet<double>;
// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

using Scalar = double;
using Index = int;
using Pair = std::pair<Index, Scalar>;
using IndVec = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
using IndMat = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Data = Tri>
Scalar get_value(const Data& x)
{
    return x.value();
}

template <>
Scalar get_value<Pair>(const Pair& x)
{
    return x.second;
}

// Partition data into two parts
//     x_0, ..., x_{k-1} | x_k, ..., x_{n-1}
// We require that for any i in {0, ..., k-1} and j in {k, ..., n-1},
//     x_i <= x_j
template <typename Data>
void tri_select(Data* x, Index n, Index k)
{
    // Early exit if n <= 0 or k <= 0 or k >= n
    if (n <= 0 || k <= 0 || k >= n)
        return;

    std::nth_element(
        x, x + k, x + n,
        [](const Data& a, const Data& b) {
            return get_value(a) < get_value(b);
        });
}

// Sort data vector
template <typename Data = Tri>
void tri_sort(Data* x, Index n)
{
    // Early exit if n <= 0
    if (n <= 0)
        return;

    std::sort(
        x, x + n,
        [](const Data& a, const Data& b) {
            return get_value(a) < get_value(b);
        });
}

// 1. First do tri_select()
// 2. If x_0 + ... + x_{k-1} <= target, return k
// 3. Otherwise, partially sort x such that
//        x_0 <= ... <= x_{k-1}
// 4. Find J such that
//        x_0 + ... + x_{J-1} <= target < x_0 + ... + x_J
template <typename Data>
Index tri_select_and_find(
    Data* x, Index n, Index k, Scalar target, Scalar& block_sum
)
{
    // Step 1
    tri_select(x, n, k);
    // Step 2
    block_sum = Scalar(0);
    for (Index i = 0; i < k; i++)
        block_sum += get_value(x[i]);
    if (block_sum <= target)
        return k;
    // Step 3
    tri_sort(x, k);
    // Step 4
    Scalar sum = Scalar(0);
    for (Index i = 0; i < k; i++)
    {
        sum += get_value(x[i]);
        if (sum > target)
            return i;
    }
    return k - 1;
}

// 1. First do tri_select()
// 2. If x_k + ... + x_{n-1} < target, return k-1
// 3. Otherwise, partially sort x such that
//        x_k <= ... <= x_{n-1}
// 4. Find J such that
//        x_{J+1} + ... + x_{n-1} < target <= x_J + ... + x_{n-1}
template <typename Data>
Index tri_select_and_find_reverse(
    Data* x, Index n, Index k, Scalar target, Scalar& block_sum
)
{
    // Step 1
    tri_select(x, n, k);
    // Step 2
    block_sum = Scalar(0);
    for (Index i = n - 1; i >= k; i--)
        block_sum += get_value(x[i]);
    if (block_sum < target)
        return k - 1;
    // Step 3
    tri_sort(x + k, n - k);
    // Step 4
    Scalar sum = Scalar(0);
    for (Index i = n - 1; i >= k; i--)
    {
        sum += get_value(x[i]);
        if (sum >= target)
            return i;
    }
    return k;
}

// Given a data array x = [x_0, x_1, ..., x_{n-1}] and a number delta,
// sort x such that {x_0, ..., x_L} are the smallest (L+1) elements of x,
// and x_0 + ... + x_{L-1} <= delta < x_0 + ... + x_L
//
// This function call returns L
//
// If sum(x) <= delta, then return L=n
//
template <typename Data>
Index find_small(Data* x, Index n, Scalar delta)
{
    // Block size
    Index m = Index(n / 32);
    m = std::max(m, Index(32));
    m = std::min(m, n);

    // Search from the left (smallest)
    // start-----working---working+bs-----end
    Data* start = x;
    Data* end = start + n;
    Data* working = start;
    Data* found = end;
    Scalar target = delta;
    for (; working < end; working += m)
    {
        Scalar block_sum;
        // The length of the array to do arg_select()
        Index len = std::distance(working, end);
        // Actual block size if <= m
        Index bs = std::min(len, m);
        Index L = tri_select_and_find(
            working, len, bs, target, block_sum);
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

// Given a data array x = [x_0, x_1, ..., x_{n-1}] and a number delta,
// sort x such that {x_0, ..., x_L} are the smallest (L+1) elements of x,
// and x_0 + ... + x_{L-1} <= delta < x_0 + ... + x_L
//
// This function call returns L
//
// If sum(x) <= delta, then return L=0
//
// We search from the right (largest), target = sum(x) - delta
//
template <typename Data>
Index find_large(Data* x, Index n, Scalar target)
{
    // Block size
    Index m = Index(n / 32);
    m = std::max(m, Index(32));
    m = std::min(m, n);

    // Search from the right (largest)
    // start-----working-bs---working-----end
    Data* start = x;
    Data* end = start + n;
    Data* working = end;
    Data* found = start;
    for (; working >= start; working -= m)
    {
        Scalar block_sum;
        // The length of the array to do arg_select()
        Index len = std::distance(start, working);
        // Actual block size if <= m
        Index bs = std::min(len, m);
        Index k = std::distance(start, working - bs);
        Index L = tri_select_and_find_reverse(
            start, len, k, target, block_sum);
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
SpMat sparsify_mat2(const Matrix& T, double delta, double density_hint)
{
    TimePoint clock_t1 = Clock::now();

    // Dimensions
    const Index n = T.rows();
    const Index m = T.cols();

    // COO form of T, with last column removed
    std::vector<Tri> T_tri(n * (m - 1));
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        const double* src = T.data() + offset;
        Tri* dest = T_tri.data() + offset;
        for (Index i = 0; i < n; i++)
        {
            dest[i] = Tri(i, j, src[i]);
        }
    }

    TimePoint clock_t2 = Clock::now();

    // Vector to store the partition index L
    IndVec L_col(m - 1);

    // Loop over columns
    #pragma omp parallel for schedule(static)
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        Tri* data = T_tri.data() + offset;
        // Search from the smallest values
        if (density_hint > 0.1)
        {
            L_col[j] = find_small(data, n, delta);
        } else {
            // Search from the largest values
            Scalar sum = T.col(j).sum();
            Scalar target = sum - delta;
            L_col[j] = find_large(data, n, target);
        }
    }

    TimePoint clock_t3 = Clock::now();

    // Vector of small values by row
    std::vector<std::vector<Pair>> small_val_row(n);
    for (Index i = 0; i < n; i++)
        small_val_row[i].reserve(Index(0.1 * m));

    // Add small values to small_val_row, and add large values to
    // sparse matrix triplet
    std::vector<Tri> tri_list;
    tri_list.reserve(Index(0.1 * n * m));
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        Tri* data = T_tri.data() + offset;
        // Small values
        Index L = L_col[j];
        for (Index l = 0; l < L; l++)
        {
            Tri val = data[l];
            Index i = val.row();
            small_val_row[i].emplace_back(j, val.value());
        }
        // Large values
        tri_list.insert(tri_list.end(), data + L, data + n);
    }

    // No longer need T_tri, so free some memory
    T_tri.clear();

    TimePoint clock_t4 = Clock::now();

    // Find large elements in the small value vector
    // Control the rowwise error
    for (Index i = 0; i < n; i++)
    {
        Pair* data = small_val_row[i].data();
        Index len = small_val_row[i].size();
        // Test row sum
        Scalar rowsum = 0.0;
        for (Index l = 0; l < len; l++)
        {
            rowsum += data[l].second;
        }
        // For simplicity of theoretical analysis we use > delta,
        // but it is OK to multiply delta with a constant to save computation
        if (rowsum > 2.0 * delta)
        {
            Scalar target = rowsum - 2.0 * delta;
            Index L = find_large(data, len, target);

            // Add large values to the sparse matrix triplet
            for (Index l = L; l < len; l++)
            {
                Pair val = data[l];
                tri_list.emplace_back(i, val.first, val.second);
            }
        }
    }

    TimePoint clock_t5 = Clock::now();

    SpMat sp(n, m - 1);
    sp.setFromTriplets(tri_list.begin(), tri_list.end());

    TimePoint clock_t6 = Clock::now();
    std::cout << "t2 - t1 = " << (clock_t2 - clock_t1).count() <<
        ", t3 - t2 = " << (clock_t3 - clock_t2).count() <<
        ", t4 - t3 = " << (clock_t4 - clock_t3).count() << std::endl;
    std::cout << "t5 - t4 = " << (clock_t5 - clock_t4).count() <<
        ", t6 - t5 = " << (clock_t6 - clock_t5).count() << std::endl;

    // Matrix D = T.leftCols(m - 1) - sp;
    // std::cout << "delta = " << delta <<
    //     ", rowdelta = " << D.rowwise().sum().maxCoeff() <<
    //     ", coldelta = " << D.colwise().sum().maxCoeff() << std::endl;
    return sp;
}


}  // namespace Sinkhorn
