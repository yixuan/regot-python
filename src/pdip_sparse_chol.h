#ifndef REGOT_PDIP_SPARSE_CHOL_H
#define REGOT_PDIP_SPARSE_CHOL_H

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace PDIP {

// PDIP-FP B2 sparse block layout (same as in pdip_fp.cpp) for assembling sparse Cholesky.
struct BlockB2 {
    int M{0};
    int n12{0};
    int N{0};
    std::vector<double> B11;
    std::vector<double> B22;
    std::vector<double> A12;
    std::vector<int> B12_row_ptr;
    std::vector<int> B12_col_idx;
    std::vector<double> B12_val;
};

// Eigen SimplicialLLT (AMD ordering): factor and solve the lower-triangular sparse SPD from the B2 path only.
class PdipSparseChol {
public:
    using SpMat = Eigen::SparseMatrix<double>;
    using SpChol = Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>>;

    bool factor(const BlockB2& blk);
    void solve(const double* b, double* x) const;

    bool ok() const { return ok_; }
    int dim() const { return N_; }

private:
    SpChol solver_;
    SpMat mat_;
    int N_{0};
    bool ok_{false};
    std::vector<Eigen::Triplet<double>> tri_buf_;
};

}  // namespace PDIP

#endif  // REGOT_PDIP_SPARSE_CHOL_H
