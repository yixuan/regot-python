// Role of this file:
// 1) PDIP-FP kernel (same code path as the first version);
// 2) Unified regot primary fields;
// 3) Optional FP per-phase timing in PDIPResult only when built with REGOT_PDIP_DEV (see pdip_dev_flags.h).
#include "pdip_solvers.h"
#include "pdip_dev_flags.h"
#include "pdip_sparse_chol.h"
#include "pdip_transport_ops.hpp"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

namespace PDIP {

using Vector = Eigen::VectorXd;
using MatRowMajor = transport::MatRowMajor;

namespace detail {

bool pdip_cholesky_lower(int n, std::vector<double>& A) {
    using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMajorMat> mat(A.data(), n, n);
    Eigen::LLT<RowMajorMat> llt(mat);
    if (llt.info() != Eigen::Success) {
        return false;
    }
    mat.template triangularView<Eigen::Upper>().setZero();
    mat.template triangularView<Eigen::Lower>() = llt.matrixL();
    return true;
}

void pdip_cholesky_solve(int n, const std::vector<double>& L, const double* b, double* x, double* work) {
    (void)work;
    using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMajorMat> Lmat(L.data(), n, n);
    Eigen::Map<const Vector> bvec(b, n);
    Eigen::Map<Vector> xvec(x, n);
    xvec = Lmat.transpose().template triangularView<Eigen::Upper>().solve(
        Lmat.template triangularView<Eigen::Lower>().solve(bvec)
    );
}

void pdip_cholesky_solve(int n, const std::vector<double>& L, const double* b, double* x) {
    std::vector<double> work(static_cast<size_t>(n));
    pdip_cholesky_solve(n, L, b, x, work.data());
}

}  // namespace detail

static const double SMALL = 1e-50;
using Clock = std::chrono::steady_clock;

#ifdef REGOT_PDIP_DEV
struct PdipFpProfile {
    double build_B_sec{0};
    double cholesky_factor_sec{0};
    double cholesky_solve_sec{0};
    double fixed_point_sec{0};
    double eq_matvec_sec{0};
    int cholesky_solve_calls{0};
    int fixed_point_calls{0};
};
#endif

static const std::vector<double>& factorize_with_fallback(
    std::vector<double>& B_dense,
    std::vector<double>& B_backup,
    std::vector<double>& B_chol_work,
    int N_block
) {
    if (detail::pdip_cholesky_lower(N_block, B_dense)) {
        return B_dense;
    }
    B_backup = B_dense;
    B_chol_work = B_backup;
    for (int shift_try = 1; shift_try < 4; ++shift_try) {
        double shift = (shift_try == 1 ? 1e-12 : (shift_try == 2 ? 1e-10 : 1e-8));
        B_chol_work = B_backup;
        for (int i = 0; i < N_block; ++i) B_chol_work[i * N_block + i] += shift;
        if (detail::pdip_cholesky_lower(N_block, B_chol_work)) {
            return B_chol_work;
        }
    }
    B_chol_work = B_backup;
    for (int i = 0; i < N_block; ++i) B_chol_work[i * N_block + i] += 1e-8;
    detail::pdip_cholesky_lower(N_block, B_chol_work);
    return B_chol_work;
}

static void eq_matvec(const double* scale, const double* x, int n, int m, double* y) {
    transport::A_matvec_from_x(n, m, x, scale, y);
}

static void eq_matvec_trans(const double* lambda, int n, int m, double* y) {
    transport::AT_matvec(n, m, lambda, y);
}

static void solve_AAT(const double* rhs, int n, int m, double* out) {
    transport::solve_AAT(rhs, n, m, out);
}

static double step_size(int n, const double* x, const double* dx) {
    Eigen::Map<const Vector> xv(x, n), dvd(dx, n);
    Eigen::ArrayXd cand = (dvd.array() < 0).select(-xv.array() / (dvd.array() + SMALL), 1.0);
    return std::min(1.0, cand.minCoeff());
}

static void build_B1_full(int M, int n, const double* scale, std::vector<double>& B_dense) {
    const int n12 = n - 1, N = M + n12;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> S(scale, M, n);
    Vector B11 = S.rowwise().sum();
    Eigen::RowVectorXd B22_row = S.leftCols(n12).colwise().sum();
    B_dense.assign(static_cast<size_t>(N * N), 0);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(B_dense.data(), N, N);
    for (int i = 0; i < M; ++i) B(i, i) = B11(i);
    for (int j = 0; j < n12; ++j) B(M + j, M + j) = B22_row(j);
    B.block(0, M, M, n12) = S.leftCols(n12);
    B.block(M, 0, n12, M) = S.leftCols(n12).transpose();
}

static void matvec_A_block(const BlockB2& blk, const double* x, double* y) {
    const int M = blk.M, n12 = blk.n12;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A12(blk.A12.data(), M, n12);
    Eigen::Map<const Vector> x1(x, M), x2(x + M, n12);
    Eigen::Map<const Vector> B11v(blk.B11.data(), M);
    Eigen::Map<const Vector> B22v(blk.B22.data(), n12);
    Eigen::Map<Vector> y1(y, M), y2(y + M, n12);
    y1.noalias() = B11v.cwiseProduct(x1) + A12 * x2;
    y2.noalias() = B22v.cwiseProduct(x2) + A12.transpose() * x1;
}

static void matvec_B_block(const BlockB2& blk, const double* x, double* y) {
    const int M = blk.M, n12 = blk.n12;
    Eigen::Map<Vector> y1(y, M), y2(y + M, n12);
    Eigen::Map<const Vector> x1(x, M), x2(x + M, n12);
    Eigen::Map<const Vector> B11v(blk.B11.data(), M);
    Eigen::Map<const Vector> B22v(blk.B22.data(), n12);
    y1.noalias() = B11v.cwiseProduct(x1);
    y2.noalias() = B22v.cwiseProduct(x2);
    for (int i = 0; i < M; ++i) {
        for (int p = blk.B12_row_ptr[i]; p < blk.B12_row_ptr[i + 1]; ++p)
            y1(i) += blk.B12_val[p] * x2(blk.B12_col_idx[p]);
    }
    for (int i = 0; i < M; ++i) {
        for (int p = blk.B12_row_ptr[i]; p < blk.B12_row_ptr[i + 1]; ++p)
            y2(blk.B12_col_idx[p]) += blk.B12_val[p] * x1(i);
    }
}

static double marginal_error(int n, int m, const double* x, const double* eq_vector) {
    Eigen::Map<const Vector> eq(eq_vector, m + n - 1);
    Eigen::Map<const MatRowMajor> X(x, n, m);
    Vector row_s = X.rowwise().sum();
    Vector a_marg(n);
    a_marg.head(n - 1) = eq.segment(m, n - 1);
    a_marg(n - 1) = 1.0 - eq.segment(m, n - 1).sum();
    const double err1 = (row_s - a_marg).norm();
    Vector col_s = X.colwise().sum().transpose();
    const double err2 = (col_s - eq.head(m)).norm();
    return std::max(err1, err2);
}

static void build_B2_fixed(int M, int n, const double* scale, double fixed_threshold,
                           std::vector<double>& B_dense, BlockB2& block, bool fill_dense) {
    const int n12 = n - 1, N = M + n12;
    block.M = M; block.n12 = n12; block.N = N;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> S(scale, M, n);
    block.B11.resize(static_cast<size_t>(M));
    block.B22.resize(static_cast<size_t>(n12));
    Eigen::Map<Vector> B11m(block.B11.data(), M);
    Eigen::Map<Vector> B22m(block.B22.data(), n12);
    B11m = S.rowwise().sum();
    B22m = S.leftCols(n12).colwise().sum().transpose();
    std::vector<int> nnz_per_row(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) {
        int cnt = 0;
        for (int j = 0; j < n12; ++j)
            if (scale[i + j * M] >= fixed_threshold) ++cnt;
        nnz_per_row[static_cast<size_t>(i)] = cnt;
    }
    block.B12_row_ptr.resize(static_cast<size_t>(M + 1));
    block.B12_row_ptr[0] = 0;
    for (int i = 0; i < M; ++i)
        block.B12_row_ptr[static_cast<size_t>(i + 1)] = block.B12_row_ptr[static_cast<size_t>(i)] + nnz_per_row[static_cast<size_t>(i)];
    const int total_nnz = block.B12_row_ptr[static_cast<size_t>(M)];
    block.B12_col_idx.resize(static_cast<size_t>(total_nnz));
    block.B12_val.resize(static_cast<size_t>(total_nnz));
    block.A12.resize(static_cast<size_t>(M * n12));
    for (int i = 0; i < M; ++i) {
        int k = block.B12_row_ptr[static_cast<size_t>(i)];
        for (int j = 0; j < n12; ++j) {
            double v = scale[i + j * M];
            block.A12[static_cast<size_t>(i * n12 + j)] = v;
            if (v >= fixed_threshold) {
                block.B12_col_idx[static_cast<size_t>(k)] = j;
                block.B12_val[static_cast<size_t>(k)] = v;
                ++k;
            }
        }
    }
    if (fill_dense) {
        B_dense.assign(static_cast<size_t>(N * N), 0);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(B_dense.data(), N, N);
        for (int i = 0; i < M; ++i) B(i, i) = block.B11[static_cast<size_t>(i)];
        for (int j = 0; j < n12; ++j) B(M + j, M + j) = block.B22[static_cast<size_t>(j)];
        for (int i = 0; i < M; ++i) {
            for (int p = block.B12_row_ptr[static_cast<size_t>(i)]; p < block.B12_row_ptr[static_cast<size_t>(i + 1)]; ++p) {
                int jj = block.B12_col_idx[static_cast<size_t>(p)];
                double v = block.B12_val[static_cast<size_t>(p)];
                B(i, M + jj) = v;
                B(M + jj, i) = v;
            }
        }
    }
}

static double compute_b12_sparsity_after_threshold(int M, int n, const double* scale, double fixed_threshold) {
    const int n12 = n - 1;
    const long total = static_cast<long>(M) * static_cast<long>(n12);
    if (total == 0) return 1.0;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> S(scale, M, n);
    const long nnz = (S.leftCols(n12).array() >= fixed_threshold).count();
    double density = static_cast<double>(nnz) / static_cast<double>(total);
    return 1.0 - density;
}

static void fixed_point_solve_block_sparse(
    int N, const BlockB2& block, const PdipSparseChol& chol,
    const double* c, double* x_out, const double* x0,
    double exit_scale, int fp_max_iter,
    std::vector<double>& rhs_work, std::vector<double>& res_work
) {
    if (rhs_work.size() < static_cast<size_t>(N)) rhs_work.resize(static_cast<size_t>(N));
    if (res_work.size() < static_cast<size_t>(N)) res_work.resize(static_cast<size_t>(N));
    std::vector<double> x(static_cast<size_t>(N));
    std::vector<double> x_tmp(static_cast<size_t>(N));
    if (x0) {
        std::memcpy(x.data(), x0, static_cast<size_t>(N) * sizeof(double));
    } else {
        chol.solve(c, x.data());
    }
    matvec_A_block(block, x.data(), res_work.data());
    Eigen::Map<const Vector> cv(c, N);
    Eigen::Map<const Vector> rw0(res_work.data(), N);
    double r_pk = (cv - rw0).norm() + 1e-20;
    for (int k = 0; k < fp_max_iter; ++k) {
        matvec_A_block(block, x.data(), res_work.data());
        {
            Eigen::Map<Vector> rhs(rhs_work.data(), N);
            Eigen::Map<const Vector> rw_a(res_work.data(), N);
            rhs = cv - rw_a;
        }
        matvec_B_block(block, x.data(), res_work.data());
        {
            Eigen::Map<Vector> rhs(rhs_work.data(), N);
            Eigen::Map<const Vector> rw_b(res_work.data(), N);
            rhs += rw_b;
        }
        chol.solve(rhs_work.data(), x_tmp.data());
        matvec_A_block(block, x_tmp.data(), res_work.data());
        Eigen::Map<const Vector> rw(res_work.data(), N);
        double res_norm = (cv - rw).norm();
        if (res_norm < exit_scale * r_pk) {
            std::memcpy(x_out, x_tmp.data(), static_cast<size_t>(N) * sizeof(double));
            return;
        }
        r_pk = res_norm;
        std::memcpy(x.data(), x_tmp.data(), static_cast<size_t>(N) * sizeof(double));
    }
    std::memcpy(x_out, x_tmp.data(), static_cast<size_t>(N) * sizeof(double));
}

void pdip_fp_internal(
    PDIPResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const PDIPSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
) {
    (void)verbose;
    (void)cout;
    // Dimensions
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    const int n_vars = n * m, n_constraints = m + n - 1, N_block = n_constraints;
    double reg_val = (reg > 0) ? reg : 1e-6;
    double barrier = reg_val;
    double fixed_threshold = opts.fixed_threshold;
    int fp_max_iter = opts.fp_max_iter;
    double exit_scale = opts.fp_exit_scale;
    if (exit_scale <= 0) exit_scale = 1e-2;

    auto t_start = Clock::now();
#ifdef REGOT_PDIP_DEV
    PdipFpProfile prof{};
    auto elapsed = [](Clock::time_point from) {
        return std::chrono::duration<double>(Clock::now() - from).count();
    };
    Clock::time_point t_prof;
#endif

    std::vector<double> cost(static_cast<size_t>(n_vars)), eq_vector(static_cast<size_t>(n_constraints));
    std::memcpy(cost.data(), M.data(), sizeof(double) * static_cast<size_t>(n_vars));
    {
        Eigen::Map<Vector> ev(eq_vector.data(), n_constraints);
        ev.head(m) = b;
        ev.segment(m, n - 1) = a.head(n - 1);
    }

    std::vector<double> scale(static_cast<size_t>(n_vars)), b1_scaled(static_cast<size_t>(n_vars));
    std::vector<double> x(static_cast<size_t>(n_vars)), s(static_cast<size_t>(n_vars)), lambda_val(static_cast<size_t>(n_constraints));
    std::vector<double> delta_x(static_cast<size_t>(n_vars)), delta_lambda(static_cast<size_t>(n_constraints)), delta_s(static_cast<size_t>(n_vars));
    std::vector<double> delta_lambda_final(static_cast<size_t>(n_constraints));
    std::vector<double> b1(static_cast<size_t>(n_vars)), b2(static_cast<size_t>(n_constraints)), c(static_cast<size_t>(n_constraints));
    std::vector<double> r_dual(static_cast<size_t>(n_vars)), r_pri(static_cast<size_t>(n_constraints));
    std::vector<double> rhs_fp(static_cast<size_t>(n_constraints)), res_fp(static_cast<size_t>(n_constraints));

    // Phase 1: initial feasible point
    std::vector<double> initial_x(static_cast<size_t>(n_vars));
    solve_AAT(eq_vector.data(), n, m, lambda_val.data());
    eq_matvec_trans(lambda_val.data(), n, m, initial_x.data());
    std::vector<double> rhs_init(static_cast<size_t>(n_constraints));
    eq_matvec(nullptr, initial_x.data(), n, m, rhs_init.data());
    eq_matvec(nullptr, cost.data(), n, m, b1.data());
    for (int i = 0; i < n_constraints; ++i) rhs_init[static_cast<size_t>(i)] = -barrier * rhs_init[static_cast<size_t>(i)] - b1[static_cast<size_t>(i)];
    solve_AAT(rhs_init.data(), n, m, lambda_val.data());
    eq_matvec_trans(lambda_val.data(), n, m, b1.data());
    std::vector<double> initial_s(static_cast<size_t>(n_vars));
    {
        Eigen::Map<Vector> ix(initial_x.data(), n_vars), is(initial_s.data(), n_vars);
        Eigen::Map<const Vector> co(cost.data(), n_vars), b1v(b1.data(), n_vars);
        is.array() = barrier * ix.array() + co.array() + b1v.array();
    }
    double dx_shift = 0, ds_shift = 0;
    for (int i = 0; i < n_vars; ++i) {
        if (initial_x[static_cast<size_t>(i)] < 0) dx_shift = std::max(dx_shift, -1.5 * initial_x[static_cast<size_t>(i)]);
        if (initial_s[static_cast<size_t>(i)] < 0) ds_shift = std::max(ds_shift, -1.5 * initial_s[static_cast<size_t>(i)]);
    }
    {
        Eigen::Map<Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
        Eigen::Map<const Vector> ix(initial_x.data(), n_vars), is(initial_s.data(), n_vars);
        xv.array() = ix.array() + dx_shift;
        sv.array() = is.array() + ds_shift;
    }
    {
        Eigen::Map<Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
        double xs = xv.dot(sv), ss = sv.sum(), xx = xv.sum();
        xv.array() += 0.5 * xs / ss;
        xs = xv.dot(sv);
        xx += xv.sum();
        sv.array() += 0.5 * xs / xx;
        xv = xv.cwiseMax(1e-10);
        sv = sv.cwiseMax(1e-10);
    }

    std::vector<double> delta_lambda_prev;
    std::vector<double> B_dense;
    BlockB2 block_b2;
    std::vector<double> mar_err_history;
    mar_err_history.reserve(static_cast<size_t>(max_iter));
    std::vector<double> obj_vals;
    obj_vals.reserve(static_cast<size_t>(max_iter));
    std::vector<double> run_times;
    run_times.reserve(static_cast<size_t>(max_iter));
    std::vector<double> B_backup;
    std::vector<double> B_chol_work;
    std::vector<double> r_c_cor(static_cast<size_t>(n_vars)), delta_s_final(static_cast<size_t>(n_vars)), delta_x_final(static_cast<size_t>(n_vars));

    // Phase 2: outer iterations
    for (int iteration = 0; iteration < max_iter; ++iteration) {
#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        {
            Eigen::Map<Vector> sc(scale.data(), n_vars);
            Eigen::Map<const Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
            sc.array() = xv.array() / (barrier * xv.array() + sv.array() + SMALL);
        }
        eq_matvec_trans(lambda_val.data(), n, m, b1.data());
        {
            Eigen::Map<Vector> rd(r_dual.data(), n_vars);
            Eigen::Map<const Vector> xv(x.data(), n_vars), cv(cost.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> b1v(b1.data(), n_vars);
            rd.array() = barrier * xv.array() + cv.array() + b1v.array() - sv.array();
        }
        eq_matvec(nullptr, x.data(), n, m, r_pri.data());
#ifdef REGOT_PDIP_DEV
        prof.eq_matvec_sec += elapsed(t_prof);
#endif
        {
            Eigen::Map<Vector> rp(r_pri.data(), n_constraints);
            Eigen::Map<const Vector> ev(eq_vector.data(), n_constraints);
            rp.head(m) -= ev.head(m);
            rp.segment(m, n - 1) -= ev.segment(m, n - 1);
        }

        bool use_fp_this_iter = false;
        bool use_sparse_chol = false;
        PdipSparseChol sparse_chol;
#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        if (iteration < 3) {
            build_B1_full(m, n, scale.data(), B_dense);
        } else {
            double sparsity = compute_b12_sparsity_after_threshold(m, n, scale.data(), fixed_threshold);
            const double fp_sparsity_threshold = 0.99;
            use_fp_this_iter = (fixed_threshold > 0.0) && (sparsity >= fp_sparsity_threshold);
            if (use_fp_this_iter) {
                build_B2_fixed(m, n, scale.data(), fixed_threshold, B_dense, block_b2, false);
                use_sparse_chol = sparse_chol.factor(block_b2);
                if (!use_sparse_chol) {
                    build_B1_full(m, n, scale.data(), B_dense);
                    use_fp_this_iter = false;
                }
            } else {
                build_B1_full(m, n, scale.data(), B_dense);
            }
        }
#ifdef REGOT_PDIP_DEV
        prof.build_B_sec += elapsed(t_prof);
#endif

#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        const std::vector<double>* B_chol_ptr = nullptr;
        if (!use_sparse_chol) {
            B_chol_ptr = &factorize_with_fallback(B_dense, B_backup, B_chol_work, N_block);
        }
#ifdef REGOT_PDIP_DEV
        prof.cholesky_factor_sec += elapsed(t_prof);
#endif

        {
            Eigen::Map<Vector> b1v(b1.data(), n_vars);
            Eigen::Map<const Vector> sv(s.data(), n_vars), rd(r_dual.data(), n_vars);
            b1v.array() = -sv.array() - rd.array();
        }
        {
            Eigen::Map<Vector> b2v(b2.data(), n_constraints);
            Eigen::Map<const Vector> rp(r_pri.data(), n_constraints);
            b2v = -rp;
        }
        {
            Eigen::Map<Vector> b1s(b1_scaled.data(), n_vars);
            Eigen::Map<const Vector> sc(scale.data(), n_vars), b1v(b1.data(), n_vars);
            b1s.array() = sc.array() * b1v.array();
        }
#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        eq_matvec(nullptr, b1_scaled.data(), n, m, c.data());
#ifdef REGOT_PDIP_DEV
        prof.eq_matvec_sec += elapsed(t_prof);
#endif
        {
            Eigen::Map<Vector> cv(c.data(), n_constraints);
            Eigen::Map<const Vector> b2v(b2.data(), n_constraints);
            cv -= b2v;
        }

        const double* x0_fp = delta_lambda_prev.empty() ? nullptr : delta_lambda_prev.data();
        if (!use_fp_this_iter) {
#ifdef REGOT_PDIP_DEV
            t_prof = Clock::now();
#endif
            detail::pdip_cholesky_solve(N_block, *B_chol_ptr, c.data(), delta_lambda.data(), res_fp.data());
#ifdef REGOT_PDIP_DEV
            prof.cholesky_solve_sec += elapsed(t_prof);
            prof.cholesky_solve_calls += 1;
#endif
        } else {
#ifdef REGOT_PDIP_DEV
            t_prof = Clock::now();
#endif
            fixed_point_solve_block_sparse(N_block, block_b2, sparse_chol, c.data(), delta_lambda.data(), x0_fp,
                                           exit_scale, fp_max_iter, rhs_fp, res_fp);
#ifdef REGOT_PDIP_DEV
            prof.fixed_point_sec += elapsed(t_prof);
            prof.fixed_point_calls += 1;
#endif
        }

#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        eq_matvec_trans(delta_lambda.data(), n, m, b1.data());
#ifdef REGOT_PDIP_DEV
        prof.eq_matvec_sec += elapsed(t_prof);
#endif
        {
            Eigen::Map<Vector> ds(delta_s.data(), n_vars), dx(delta_x.data(), n_vars);
            Eigen::Map<const Vector> sc(scale.data(), n_vars), rd(r_dual.data(), n_vars), xv(x.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> b1v(b1.data(), n_vars);
            ds.array() = (1.0 - barrier * sc.array()) * (b1v.array() + rd.array() - barrier * xv.array());
            dx.array() = (-xv.array() * sv.array() - xv.array() * ds.array()) / (sv.array() + SMALL);
        }

        double alpha1 = std::min(step_size(n_vars, x.data(), delta_x.data()), 1.0);
        alpha1 = std::min(alpha1, step_size(n_vars, s.data(), delta_s.data()));
        alpha1 = std::min(alpha1, 1.0);
        double mu = 0;
        {
            Eigen::Map<const Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
            mu = xv.dot(sv) / static_cast<double>(n_vars);
        }
        double mu_aff = 0;
        {
            Eigen::Map<const Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> dxv(delta_x.data(), n_vars), dsv(delta_s.data(), n_vars);
            mu_aff = ((xv.array() + alpha1 * dxv.array()) * (sv.array() + alpha1 * dsv.array())).sum() / static_cast<double>(n_vars);
        }
        double sigma = std::min(std::pow(mu_aff / mu, 3.0), 1.0);
        {
            Eigen::Map<Vector> rcc(r_c_cor.data(), n_vars);
            Eigen::Map<const Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> dxv(delta_x.data(), n_vars), dsv(delta_s.data(), n_vars);
            rcc.array() = xv.array() * sv.array() + dxv.array() * dsv.array() - sigma * mu;
        }
        {
            Eigen::Map<Vector> b1v(b1.data(), n_vars);
            Eigen::Map<const Vector> rcc(r_c_cor.data(), n_vars), rd(r_dual.data(), n_vars), xv(x.data(), n_vars);
            b1v.array() = -rcc.array() / (xv.array() + SMALL) - rd.array();
        }
        {
            Eigen::Map<Vector> b2v(b2.data(), n_constraints);
            Eigen::Map<const Vector> rp(r_pri.data(), n_constraints);
            b2v = -rp;
        }
        {
            Eigen::Map<Vector> b1s(b1_scaled.data(), n_vars);
            Eigen::Map<const Vector> sc(scale.data(), n_vars), b1v(b1.data(), n_vars);
            b1s.array() = sc.array() * b1v.array();
        }
#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        eq_matvec(nullptr, b1_scaled.data(), n, m, c.data());
#ifdef REGOT_PDIP_DEV
        prof.eq_matvec_sec += elapsed(t_prof);
#endif
        {
            Eigen::Map<Vector> cv(c.data(), n_constraints);
            Eigen::Map<const Vector> b2v(b2.data(), n_constraints);
            cv -= b2v;
        }
        if (!use_fp_this_iter) {
#ifdef REGOT_PDIP_DEV
            t_prof = Clock::now();
#endif
            detail::pdip_cholesky_solve(N_block, *B_chol_ptr, c.data(), delta_lambda_final.data(), res_fp.data());
#ifdef REGOT_PDIP_DEV
            prof.cholesky_solve_sec += elapsed(t_prof);
            prof.cholesky_solve_calls += 1;
#endif
        } else {
#ifdef REGOT_PDIP_DEV
            t_prof = Clock::now();
#endif
            fixed_point_solve_block_sparse(N_block, block_b2, sparse_chol, c.data(), delta_lambda_final.data(),
                                           delta_lambda.data(), exit_scale, fp_max_iter, rhs_fp, res_fp);
#ifdef REGOT_PDIP_DEV
            prof.fixed_point_sec += elapsed(t_prof);
            prof.fixed_point_calls += 1;
#endif
        }

#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        eq_matvec_trans(delta_lambda_final.data(), n, m, b1.data());
#ifdef REGOT_PDIP_DEV
        prof.eq_matvec_sec += elapsed(t_prof);
#endif
        {
            Eigen::Map<Vector> dsf(delta_s_final.data(), n_vars), dxf(delta_x_final.data(), n_vars);
            Eigen::Map<const Vector> sc(scale.data(), n_vars), rd(r_dual.data(), n_vars), xv(x.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> rcc(r_c_cor.data(), n_vars);
            Eigen::Map<const Vector> b1v(b1.data(), n_vars);
            dsf.array() = (1.0 - barrier * sc.array()) * (b1v.array() + rd.array() - barrier * rcc.array() / (sv.array() + SMALL));
            dxf.array() = (-rcc.array() - xv.array() * dsf.array()) / (sv.array() + SMALL);
        }
        double alpha2 = std::min(step_size(n_vars, x.data(), delta_x_final.data()), step_size(n_vars, s.data(), delta_s_final.data()));
        alpha2 = std::min(0.99 * alpha2, 1.0);
        {
            Eigen::Map<Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> dxf(delta_x_final.data(), n_vars), dsf(delta_s_final.data(), n_vars);
            xv.noalias() += alpha2 * dxf;
            sv.noalias() += alpha2 * dsf;
        }
        {
            Eigen::Map<Vector> lv(lambda_val.data(), n_constraints);
            Eigen::Map<const Vector> dlf(delta_lambda_final.data(), n_constraints);
            lv.noalias() += alpha2 * dlf;
        }
        delta_lambda_prev = delta_lambda_final;

        const double current_mar_err = marginal_error(n, m, x.data(), eq_vector.data());
        mar_err_history.push_back(current_mar_err);
        double obj = 0;
        {
            Eigen::Map<const Vector> xv(x.data(), n_vars), cv(cost.data(), n_vars);
            obj = xv.dot(cv) + (reg_val / 2.0) * xv.squaredNorm();
        }
        obj_vals.push_back(obj);
        run_times.push_back(std::chrono::duration<double>(Clock::now() - t_start).count() * 1000.0);

#ifdef REGOT_PDIP_DEV
        t_prof = Clock::now();
#endif
        eq_matvec_trans(lambda_val.data(), n, m, b1.data());
        {
            Eigen::Map<Vector> rd(r_dual.data(), n_vars);
            Eigen::Map<const Vector> xv(x.data(), n_vars), cv(cost.data(), n_vars), sv(s.data(), n_vars);
            Eigen::Map<const Vector> b1v(b1.data(), n_vars);
            rd.array() = barrier * xv.array() + cv.array() + b1v.array() - sv.array();
        }
        eq_matvec(nullptr, x.data(), n, m, r_pri.data());
#ifdef REGOT_PDIP_DEV
        prof.eq_matvec_sec += elapsed(t_prof);
#endif
        {
            Eigen::Map<Vector> rp(r_pri.data(), n_constraints);
            Eigen::Map<const Vector> ev(eq_vector.data(), n_constraints);
            rp.head(m) -= ev.head(m);
            rp.segment(m, n - 1) -= ev.segment(m, n - 1);
        }
        double dual_gap = 0, primal_gap = 0;
        {
            Eigen::Map<const Vector> rd(r_dual.data(), n_vars);
            dual_gap = rd.norm();
        }
        {
            Eigen::Map<const Vector> rp(r_pri.data(), n_constraints);
            primal_gap = rp.norm();
        }
        double norm_eq = 0;
        {
            Eigen::Map<const Vector> eqv(eq_vector.data(), n_constraints);
            norm_eq = eqv.norm();
        }
        double denom_pri = 1.0 + norm_eq;
        double norm_cost = 0, norm_at_lambda = 0;
        {
            Eigen::Map<const Vector> cv(cost.data(), n_vars), b1v(b1.data(), n_vars);
            norm_cost = cv.norm();
            norm_at_lambda = b1v.norm();
        }
        double denom_dual = 1.0 + norm_cost + norm_at_lambda;
        dual_gap /= denom_dual;
        primal_gap /= denom_pri;
        double mu_stop = 0;
        {
            Eigen::Map<const Vector> xv(x.data(), n_vars), sv(s.data(), n_vars);
            mu_stop = xv.dot(sv) / static_cast<double>(n_vars);
        }
        const bool by_gap_mu =
            (primal_gap < tol && dual_gap < tol && mu_stop < tol);
        const bool by_mar = (current_mar_err < tol);
        const bool stop = opts.fp_stop_gap_mu_only ? by_gap_mu : (by_gap_mu || by_mar);
        if (stop) {
            result.converged = true;
            result.niter = iteration + 1;
            break;
        }
        result.niter = iteration + 1;
    }

    // Phase 3: fill result
    result.plan.resize(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            result.plan(i, j) = x[static_cast<size_t>(i * m + j)];
    result.obj_vals = std::move(obj_vals);
    result.mar_errs = std::move(mar_err_history);
    result.run_times = std::move(run_times);
#ifdef REGOT_PDIP_DEV
    result.t_build_B = prof.build_B_sec;
    result.t_chol_factor = prof.cholesky_factor_sec;
    result.t_chol_solve = prof.cholesky_solve_sec;
    result.t_eq_matvec = prof.eq_matvec_sec;
    double t_total = std::chrono::duration<double>(Clock::now() - t_start).count();
    result.t_other = std::max(0.0, t_total - (result.t_build_B + result.t_chol_factor + result.t_chol_solve + result.t_eq_matvec));
#endif
}

}  // namespace PDIP
