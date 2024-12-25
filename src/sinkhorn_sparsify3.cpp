#include <iostream>
#include <numeric>  // std::accumulate
#include <chrono>
#include "sinkhorn_sparsify.h"

// Fast sorting functions
#ifdef __AVX2__
#include "sorting/xss-common-includes.h"
#include "sorting/xss-common-qsort.h"
#include "sorting/xss-common-argsort.h"
#include "sorting/avx2-64bit-qsort.hpp"
#include "sorting/avx2-32bit-half.hpp"
#endif

// Whether to print detailed timing information
// #define TIMING 1

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;

using Scalar = double;
using Index = int;

#ifdef TIMING
// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;
#endif

// 1. First do nth_element() partition such that x is reordered and
//    x_0, ..., x_{k-1} <= any element in [x_k, ..., x_{n-1}]
// 2. If x_0 + ... + x_{k-1} <= target, return k
// 3. Otherwise, partially sort x such that
//        x_0 <= ... <= x_{k-1}
// 4. Find J such that
//        x_0 + ... + x_{J-1} <= target < x_0 + ... + x_J
inline Index select_and_find(
    Scalar* x, Index n, Index k, Scalar target, Scalar& block_sum
)
{
    // Step 1
#ifdef __AVX2__
    avx2_qselect(x, k, n);
#else
    std::nth_element(x, x + k, x + n);
#endif
    // Step 2
    block_sum = std::accumulate(x, x + k, Scalar(0));
    if (block_sum <= target)
        return k;
    // Step 3
#ifdef __AVX2__
    avx2_qsort(x, k);
#else
    std::sort(x, x + k);
#endif
    // Step 4
    Scalar sum = Scalar(0);
    for (Index i = 0; i < k; i++)
    {
        sum += x[i];
        if (sum > target)
            return i;
    }
    return k - 1;
}

// 1. First do nth_element() partition such that x is reordered and
//    x_0, ..., x_{k-1} <= any element in [x_k, ..., x_{n-1}]
// 2. If x_k + ... + x_{n-1} < target, return k-1
// 3. Otherwise, partially sort x such that
//        x_k <= ... <= x_{n-1}
// 4. Find J such that
//        x_{J+1} + ... + x_{n-1} < target <= x_J + ... + x_{n-1}
inline Index select_and_find_reverse(
    Scalar* x, Index n, Index k, Scalar target, Scalar& block_sum
)
{
    // Step 1
#ifdef __AVX2__
    avx2_qselect(x, k, n);
#else
    std::nth_element(x, x + k, x + n);
#endif
    // Step 2
    block_sum = std::accumulate(x + k, x + n, Scalar(0));
    if (block_sum < target)
        return k - 1;
    // Step 3
#ifdef __AVX2__
    avx2_qsort(x + k, n - k);
#else
    std::sort(x + k, x + n);
#endif
    // Step 4
    Scalar sum = Scalar(0);
    for (Index i = n - 1; i >= k; i--)
    {
        sum += x[i];
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
inline Index find_small(Scalar* x, Index n, Scalar delta)
{
    // Block size
    Index m = Index(n / 32);
    m = std::max(m, Index(32));
    m = std::min(m, n);

    // Search from the left (smallest)
    // start-----working---working+bs-----end
    Scalar* start = x;
    Scalar* end = start + n;
    Scalar* working = start;
    Scalar* found = end;
    Scalar target = delta;
    for (; working < end; working += m)
    {
        Scalar block_sum;
        // The length of the array to do arg_select()
        Index len = std::distance(working, end);
        // Actual block size if <= m
        Index bs = std::min(len, m);
        Index L = select_and_find(
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
inline Index find_large(Scalar* x, Index n, Scalar target)
{
    // Block size
    Index m = Index(n / 32);
    m = std::max(m, Index(32));
    m = std::min(m, n);

    // Search from the right (largest)
    // start-----working-bs---working-----end
    Scalar* start = x;
    Scalar* end = start + n;
    Scalar* working = end;
    Scalar* found = start;
    for (; working >= start; working -= m)
    {
        Scalar block_sum;
        // The length of the array to do arg_select()
        Index len = std::distance(start, working);
        // Actual block size if <= m
        Index bs = std::min(len, m);
        Index k = std::distance(start, working - bs);
        Index L = select_and_find_reverse(
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

// dest[i] = if (src[i] >= thresh) 0 else src[i]
inline void apply_thresh_mask(
    const Scalar* src, Index n, Scalar thresh, Scalar* dest)
{
#ifdef __AVX2__
    // Packet type
    using Eigen::internal::ploadu;
    using Eigen::internal::pstoreu;
    using Eigen::internal::pset1;
    using Eigen::internal::pcmp_le;
    using Eigen::internal::pselect;
    using Packet = Eigen::internal::packet_traits<Scalar>::type;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Vectorized scalars
    const Packet vthresh = pset1<Packet>(thresh);
    const Packet vzero = pset1<Packet>(Scalar(0));

    // Compute for loop end points
    // n % (2^k) == n & (2^k-1), see https://stackoverflow.com/q/3072665
    // const Index peeling_end = n - n % Increment;
    const Index aligned_end = n - (n & (PacketSize - 1));
    const Index peeling_end = n - (n & (Increment - 1));

    // Working pointers
    const Scalar* wsrc = src;
    Scalar* wdest = dest;

    for (Index i = 0; i < peeling_end; i += Increment)
    {
        Packet vsrc1 = ploadu<Packet>(wsrc);
        Packet vsrc2 = ploadu<Packet>(wsrc + PacketSize);
        Packet vdest1 = pselect<Packet>(pcmp_le(vthresh, vsrc1), vzero, vsrc1);
        Packet vdest2 = pselect<Packet>(pcmp_le(vthresh, vsrc2), vzero, vsrc2);

        pstoreu(wdest, vdest1);
        pstoreu(wdest + PacketSize, vdest2);

        wsrc += Increment;
        wdest += Increment;
    }
    if (aligned_end != peeling_end)
    {
        wsrc = src + peeling_end;
        wdest = dest + peeling_end;

        Packet vsrc = ploadu<Packet>(wsrc);
        Packet vdest = pselect<Packet>(pcmp_le(vthresh, vsrc), vzero, vsrc);

        pstoreu(wdest, vdest);
    }
    // Remaining elements
    for (Index i = aligned_end; i < n; i++)
    {
        dest[i] = (src[i] >= thresh) ? Scalar(0) : src[i];
    }
#else
    const Scalar* end = src + n;
    for (; src < end; src++, dest++)
    {
        *dest = (*src >= thresh) ? Scalar(0) : (*src);
    }
#endif
}

inline void apply_thresh_mask(
    const Scalar* src, Index n, Index stride, Scalar thresh, Scalar* dest)
{
    const Scalar* end = src + stride * n;
    for (; src < end; src += stride, dest += stride)
    {
        *dest = (*src >= thresh) ? Scalar(0) : (*src);
    }
}

// Sparsify a dense matrix, with the last column removed
SpMat sparsify_mat3(const Matrix& T, double delta, double density_hint)
{
#ifdef TIMING
    TimePoint clock_t1 = Clock::now();
#endif

    // Dimensions
    const Index n = T.rows();
    const Index m = T.cols();

    // Thresholding T by columns
    Matrix Delta = T;
    Delta.col(m - 1).setZero();

#ifdef TIMING
    TimePoint clock_t2 = Clock::now();
#endif

    #pragma omp parallel for schedule(dynamic)
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        const Scalar* Tdata = T.data() + offset;
        Scalar* data = Delta.data() + offset;
        Index L = n;
        // Search from the smallest values
        if (density_hint > 0.2)
        {
            L = find_small(data, n, delta);
        } else {
            // Search from the largest values
            Scalar sum = T.col(j).sum();
            Scalar target = sum - delta;
            L = find_large(data, n, target);
        }
        // Any value >= data[L] will be set to zero
        // If L==n, then no value in data should be set to zero,
        // which is equivalent to setting thresh=Inf
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        Scalar thresh = (L >= n) ? Inf : data[L];
        // Apply the thresholding mask
        apply_thresh_mask(Tdata, n, thresh, data);
    }

#ifdef TIMING
    TimePoint clock_t3 = Clock::now();
#endif

    // Thresholding Delta by rows
    // Test row sums
    // Vector rowsum = Delta.rowwise().sum();

#ifdef TIMING
    TimePoint clock_t4 = Clock::now();
#endif

    #pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < n; i++)
    {
        Vector row = Delta.row(i).transpose();
        const Scalar rowsum = row.sum();
        // For simplicity of theoretical analysis we use > delta,
        // but it is OK to multiply delta with a constant to save computation
        if (rowsum > 2.0 * delta)
        {
            Scalar* data = row.data();
            Scalar target = rowsum - 2.0 * delta;
            Index L = find_large(data, m - 1, target);
            // Any value >= data[L] will be set to zero
            // If L==m-1, then no value in data should be set to zero,
            // which is equivalent to setting thresh=Inf
            constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
            Scalar thresh = (L >= m - 1) ? Inf : data[L];
            Scalar* Ddata = Delta.data() + i;
            apply_thresh_mask(Ddata, m - 1, n, thresh, Ddata);
        }
    }

#ifdef TIMING
    TimePoint clock_t5 = Clock::now();
#endif

    // Generate sparse matrix
    // SpMat sp = (T - Delta).leftCols(m - 1).sparseView();

    std::vector<double> value;
    std::vector<int> inner_ind;
    std::vector<int> outer_ind(m);
    outer_ind[0] = 0;
    value.reserve(int(1.1 * density_hint * n * m));
    inner_ind.reserve(int(1.1 * density_hint * n * m));
    for (Index j = 0; j < m - 1; j++)
    {
        Index offset = n * j;
        const Scalar* Tdata = T.data() + offset;
        const Scalar* Dhead = Delta.data() + offset;
        const Scalar* Dend = Dhead + n;
        const Scalar* Ddata = Dhead;
        Index nnz = 0;
        for (; Ddata < Dend; Ddata++, Tdata++)
        {
            if (*Ddata == Scalar(0))
            {
                value.emplace_back(*Tdata);
                inner_ind.emplace_back(std::distance(Dhead, Ddata));
                nnz++;
            }
        }
        outer_ind[j + 1] = outer_ind[j] + nnz;
    }
    Eigen::Map<SpMat> sp(
        n, m - 1, value.size(), outer_ind.data(), inner_ind.data(), value.data());

#ifdef TIMING
    TimePoint clock_t6 = Clock::now();
    std::cout << "[sparsify]================================================" << std::endl;
    std::cout << "mem_copy = " << (clock_t2 - clock_t1).count() <<
        ", row_sum = " << (clock_t4 - clock_t3).count() << std::endl;
    std::cout << "col_sp = " << (clock_t3 - clock_t2).count() <<
        ", row_sp = " << (clock_t5 - clock_t4).count() <<
        ", sp_mat = " << (clock_t6 - clock_t5).count() << std::endl;
    std::cout << "==========================================================" << std::endl;
#endif

    // std::cout << "delta = " << delta <<
    //     ", rowdelta = " << Delta.rowwise().sum().maxCoeff() <<
    //     ", coldelta = " << Delta.colwise().sum().maxCoeff() << std::endl;
    return sp;
}


}  // namespace Sinkhorn
