// PDIP-FP B2 path: Simplicial Cholesky (Eigen) for sparse SPD matrices, with diagonal shift fallback.
#include "pdip_sparse_chol.h"

namespace PDIP {

bool PdipSparseChol::factor(const BlockB2& blk) {
    const int M = blk.M;
    const int n12 = blk.n12;
    N_ = blk.N;
    ok_ = false;
    if (N_ <= 0 || M <= 0 || n12 < 0) {
        return false;
    }

    auto build_triplets = [&](double shift) {
        tri_buf_.clear();
        tri_buf_.reserve(static_cast<size_t>(M + n12 + 2 * blk.B12_val.size() + 64));
        for (int i = 0; i < M; ++i) {
            tri_buf_.emplace_back(i, i, blk.B11[static_cast<size_t>(i)] + shift);
        }
        for (int j = 0; j < n12; ++j) {
            tri_buf_.emplace_back(M + j, M + j, blk.B22[static_cast<size_t>(j)] + shift);
        }
        for (int i = 0; i < M; ++i) {
            for (int p = blk.B12_row_ptr[static_cast<size_t>(i)];
                 p < blk.B12_row_ptr[static_cast<size_t>(i) + 1]; ++p) {
                const int jj = blk.B12_col_idx[static_cast<size_t>(p)];
                const double v = blk.B12_val[static_cast<size_t>(p)];
                tri_buf_.emplace_back(M + jj, i, v);
            }
        }
    };

    const double shifts[] = {0.0, 1e-12, 1e-10, 1e-8};
    for (double shift : shifts) {
        build_triplets(shift);
        mat_.resize(N_, N_);
        mat_.setFromTriplets(tri_buf_.begin(), tri_buf_.end());
        mat_.makeCompressed();
        solver_.compute(mat_);
        if (solver_.info() == Eigen::Success) {
            ok_ = true;
            return true;
        }
    }
    build_triplets(1e-8);
    mat_.resize(N_, N_);
    mat_.setFromTriplets(tri_buf_.begin(), tri_buf_.end());
    mat_.makeCompressed();
    solver_.compute(mat_);
    ok_ = (solver_.info() == Eigen::Success);
    return ok_;
}

void PdipSparseChol::solve(const double* b, double* x) const {
    Eigen::Map<const Eigen::VectorXd> bv(b, N_);
    Eigen::Map<Eigen::VectorXd> xv(x, N_);
    xv = solver_.solve(bv);
}

}  // namespace PDIP
