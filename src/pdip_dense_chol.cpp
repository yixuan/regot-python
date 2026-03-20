// 文件职责：
// 为 PDIP-FP 提供稠密 Cholesky 分解与求解后端。
// 优先使用 LAPACK；否则使用 Eigen 路径，保证可读性与可移植性。
#include "pdip_dense_chol.h"
#include <cmath>
#include <cstring>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#ifdef USE_LAPACK
#include <lapacke.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

bool pdip_cholesky_lower(int n, std::vector<double>& A) {
#ifdef USE_LAPACK
    lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)n, A.data(), (lapack_int)n);
    return (info == 0);
#else
    using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMajorMat> mat(A.data(), n, n);
    Eigen::LLT<RowMajorMat> llt(mat);
    if (llt.info() != Eigen::Success) {
        return false;
    }
    mat.template triangularView<Eigen::Upper>().setZero();
    mat.template triangularView<Eigen::Lower>() = llt.matrixL();
    return true;
#endif
}

// 对外简化接口：内部自动分配工作区。
void pdip_cholesky_solve(int n, const std::vector<double>& L, const double* b, double* x) {
    std::vector<double> work(n);
    pdip_cholesky_solve(n, L, b, x, work.data());
}

void pdip_cholesky_solve(int n, const std::vector<double>& L, const double* b, double* x, double* work) {
#ifdef USE_LAPACK
    std::memcpy(x, b, (size_t)n * sizeof(double));
    lapack_int nn = (lapack_int)n;
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', nn, 1, L.data(), nn, x, 1);
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'T', 'N', nn, 1, L.data(), nn, x, 1);
#else
    using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    (void)work;
    Eigen::Map<const RowMajorMat> Lmat(L.data(), n, n);
    Eigen::Map<const Eigen::VectorXd> bvec(b, n);
    Eigen::Map<Eigen::VectorXd> xvec(x, n);
    xvec = Lmat.transpose().template triangularView<Eigen::Upper>().solve(
        Lmat.template triangularView<Eigen::Lower>().solve(bvec)
    );
#endif
}
