// 文件职责：
// 1) 实现 PDIP-FP 内核（保持第一版同源路径）；
// 2) 输出统一的 regot 主字段 + FP 分段诊断耗时；
// 3) 在不改算法语义的前提下，保留可读的阶段划分与性能统计点。
#include "pdip_solvers.h"
#include "pdip_dense_chol.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_LAPACK
extern "C" void dgemv_(const char* trans, const int* M, const int* N,
                       const double* alpha, const double* A, const int* LDA,
                       const double* x, const int* incx, const double* beta,
                       double* y, const int* incy);
extern "C" double ddot_(const int* n, const double* x, const int* incx,
                        const double* y, const int* incy);
#endif

namespace PDIP {

static const double SMALL = 1e-50;
using Clock = std::chrono::steady_clock;

struct PdipFpProfile {
    double build_B_sec{0};
    double cholesky_factor_sec{0};
    double cholesky_solve_sec{0};
    double fixed_point_sec{0};
    double eq_matvec_sec{0};
    int cholesky_solve_calls{0};
    int fixed_point_calls{0};
};

// 对称正定分解的统一入口：
// - 优先原地分解（减少大矩阵拷贝）；
// - 失败后按固定 shift 回退；
// - 返回可直接用于 solve 的下三角分解矩阵引用。
static const std::vector<double>& factorize_with_fallback(
    std::vector<double>& B_dense,
    std::vector<double>& B_backup,
    std::vector<double>& B_chol_work,
    int N_block
) {
    if (pdip_cholesky_lower(N_block, B_dense)) {
        return B_dense;
    }
    B_backup = B_dense;
    B_chol_work = B_backup;
    for (int shift_try = 1; shift_try < 4; ++shift_try) {
        double shift = (shift_try == 1 ? 1e-12 : (shift_try == 2 ? 1e-10 : 1e-8));
        B_chol_work = B_backup;
        for (int i = 0; i < N_block; ++i) B_chol_work[i * N_block + i] += shift;
        if (pdip_cholesky_lower(N_block, B_chol_work)) {
            return B_chol_work;
        }
    }
    // 保底：最大 shift 强制一次，保持行为稳定。
    B_chol_work = B_backup;
    for (int i = 0; i < N_block; ++i) B_chol_work[i * N_block + i] += 1e-8;
    pdip_cholesky_lower(N_block, B_chol_work);
    return B_chol_work;
}

static void eq_matvec(const double* scale, const double* x, int n, int m, double* y) {
#ifdef USE_LAPACK
    if (scale) {
        const int one = 1;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < m; ++j)
            y[j] = ddot_(&n, &x[j], &m, &scale[j], &m);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n - 1; ++i)
            y[m + i] = ddot_(&m, &x[i * m], &one, &scale[i * m], &one);
        return;
    }
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < m; ++j) {
        double s = 0;
        for (int i = 0; i < n; ++i) s += x[i * m + j] * (scale ? scale[i * m + j] : 1.0);
        y[j] = s;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n - 1; ++i) {
        double s = 0;
        for (int j = 0; j < m; ++j) s += x[i * m + j] * (scale ? scale[i * m + j] : 1.0);
        y[m + i] = s;
    }
}

static void eq_matvec_trans(const double* lambda, int n, int m, double* y) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        double lam_i = (i < n - 1 ? lambda[m + i] : 0);
        for (int j = 0; j < m; ++j)
            y[i * m + j] = lambda[j] + lam_i;
    }
}

static void solve_AAT(const double* rhs, int n, int m, double* out) {
    double sum_d = 0;
    for (int j = 0; j < m; ++j) sum_d += rhs[j];
    std::vector<double> tmp(n - 1);
    for (int i = 0; i < n - 1; ++i)
        tmp[i] = (sum_d / n) - rhs[m + i];
    double sum_t = 0;
    for (int i = 0; i < n - 1; ++i) sum_t += tmp[i];
    for (int i = 0; i < n - 1; ++i)
        out[m + i] = (tmp[i] + sum_t) / m;
    double sum_x2 = 0;
    for (int i = 0; i < n - 1; ++i) sum_x2 += out[m + i];
    for (int j = 0; j < m; ++j)
        out[j] = (rhs[j] - sum_x2) / n;
}

static double step_size(int n, const double* x, const double* dx) {
    double a = 1.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(min:a) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        if (dx[i] < 0) {
            double t = -x[i] / (dx[i] + SMALL);
            if (t < a) a = t;
        }
    }
    return a;
}

static void build_B1_full(int M, int n, const double* scale, std::vector<double>& B_dense) {
    const int n12 = n - 1, N = M + n12;
    std::vector<double> B11(M), B22(n12);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += scale[i + j * M];
        B11[i] = s;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n12; ++j) {
        double s = 0;
        for (int i = 0; i < M; ++i) s += scale[i + j * M];
        B22[j] = s;
    }
    B_dense.assign(N * N, 0);
    for (int i = 0; i < M; ++i) B_dense[i * N + i] = B11[i];
    for (int i = 0; i < n12; ++i) B_dense[(M + i) * N + (M + i)] = B22[i];
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < n12; ++j) {
            double v = scale[i + j * M];
            B_dense[i * N + (M + j)] = v;
            B_dense[(M + j) * N + i] = v;
        }
}

struct BlockB2 {
    int M, n12, N;
    std::vector<double> B11, B22;
    std::vector<double> A12;
    std::vector<int> B12_row_ptr;
    std::vector<int> B12_col_idx;
    std::vector<double> B12_val;
};

static void matvec_A_block(const BlockB2& blk, const double* x, double* y) {
    const int M = blk.M, n12 = blk.n12;
    const double* x1 = x;
    const double* x2 = x + M;
    double* y1 = y;
    double* y2 = y + M;
#ifdef USE_LAPACK
    const char nt = 'N', tr = 'T';
    const int one = 1;
    const double one_d = 1.0, zero_d = 0.0;
    dgemv_(&nt, &M, &n12, &one_d, blk.A12.data(), &M, x2, &one, &zero_d, y1, &one);
    for (int i = 0; i < M; ++i) y1[i] += blk.B11[i] * x1[i];
    dgemv_(&tr, &M, &n12, &one_d, blk.A12.data(), &M, x1, &one, &zero_d, y2, &one);
    for (int j = 0; j < n12; ++j) y2[j] += blk.B22[j] * x2[j];
#else
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        double t = blk.B11[i] * x1[i];
        for (int j = 0; j < n12; ++j) t += blk.A12[i * n12 + j] * x2[j];
        y1[i] = t;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n12; ++j) {
        double t = blk.B22[j] * x2[j];
        for (int i = 0; i < M; ++i) t += blk.A12[i * n12 + j] * x1[i];
        y2[j] = t;
    }
#endif
}

static void matvec_B_block(const BlockB2& blk, const double* x, double* y) {
    const int M = blk.M, n12 = blk.n12;
    const double* x1 = x;
    const double* x2 = x + M;
    double* y1 = y;
    double* y2 = y + M;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) y1[i] = blk.B11[i] * x1[i];
    for (int j = 0; j < n12; ++j) y2[j] = blk.B22[j] * x2[j];
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        for (int p = blk.B12_row_ptr[i]; p < blk.B12_row_ptr[i + 1]; ++p)
            y1[i] += blk.B12_val[p] * x2[blk.B12_col_idx[p]];
    }
    for (int i = 0; i < M; ++i) {
        for (int p = blk.B12_row_ptr[i]; p < blk.B12_row_ptr[i + 1]; ++p)
            y2[blk.B12_col_idx[p]] += blk.B12_val[p] * x1[i];
    }
}

static double marginal_error(int n, int m, const double* x, const double* eq_vector) {
    const double* b_marg = eq_vector;
    double a_sum_n1 = 0;
    for (int i = 0; i < n - 1; ++i) a_sum_n1 += eq_vector[m + i];
    double a_last = 1.0 - a_sum_n1;
    double err1 = 0, err2 = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:err1) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        double rs = 0;
        for (int j = 0; j < m; ++j) rs += x[i * m + j];
        double a_i = (i < n - 1) ? eq_vector[m + i] : a_last;
        double d = rs - a_i;
        err1 += d * d;
    }
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:err2) schedule(static)
#endif
    for (int j = 0; j < m; ++j) {
        double cs = 0;
        for (int i = 0; i < n; ++i) cs += x[i * m + j];
        double d = cs - b_marg[j];
        err2 += d * d;
    }
    err1 = std::sqrt(err1);
    err2 = std::sqrt(err2);
    return std::max(err1, err2);
}

static void build_B2_fixed(int M, int n, const double* scale, double fixed_threshold,
                           std::vector<double>& B_dense, BlockB2& block) {
    const int n12 = n - 1, N = M + n12;
    block.M = M; block.n12 = n12; block.N = N;
    block.B11.resize(M);
    block.B22.resize(n12);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += scale[i + j * M];
        block.B11[i] = s;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n12; ++j) {
        double s = 0;
        for (int i = 0; i < M; ++i) s += scale[i + j * M];
        block.B22[j] = s;
    }
    std::vector<int> nnz_per_row(M);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        int cnt = 0;
        for (int j = 0; j < n12; ++j)
            if (scale[i + j * M] >= fixed_threshold) ++cnt;
        nnz_per_row[i] = cnt;
    }
    block.B12_row_ptr.resize(M + 1);
    block.B12_row_ptr[0] = 0;
    for (int i = 0; i < M; ++i)
        block.B12_row_ptr[i + 1] = block.B12_row_ptr[i] + nnz_per_row[i];
    const int total_nnz = block.B12_row_ptr[M];
    block.B12_col_idx.resize(total_nnz);
    block.B12_val.resize(total_nnz);
    block.A12.resize(M * n12);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        int k = block.B12_row_ptr[i];
        for (int j = 0; j < n12; ++j) {
            double v = scale[i + j * M];
            block.A12[i * n12 + j] = v;
            if (v >= fixed_threshold) {
                block.B12_col_idx[k] = j;
                block.B12_val[k] = v;
                ++k;
            }
        }
    }
    B_dense.assign(N * N, 0);
    for (int i = 0; i < M; ++i) B_dense[i * N + i] = block.B11[i];
    for (int i = 0; i < n12; ++i) B_dense[(M + i) * N + (M + i)] = block.B22[i];
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        for (int p = block.B12_row_ptr[i]; p < block.B12_row_ptr[i + 1]; ++p) {
            int j = block.B12_col_idx[p];
            double v = block.B12_val[p];
            B_dense[i * N + (M + j)] = v;
            B_dense[(M + j) * N + i] = v;
        }
    }
}

static double compute_b12_sparsity_after_threshold(int M, int n, const double* scale, double fixed_threshold) {
    const int n12 = n - 1;
    const long total = static_cast<long>(M) * static_cast<long>(n12);
    if (total == 0) return 1.0;
    long nnz = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:nnz) collapse(2) schedule(static)
#endif
    for (int j = 0; j < n12; ++j)
        for (int i = 0; i < M; ++i)
            if (scale[i + j * M] >= fixed_threshold) ++nnz;
    double density = static_cast<double>(nnz) / static_cast<double>(total);
    return 1.0 - density;
}

static void fixed_point_solve_block(int N, const BlockB2& block, const std::vector<double>& B_chol,
                                    const double* c, double* x_out, const double* x0,
                                    double exit_scale, int fp_max_iter,
                                    std::vector<double>& rhs_work, std::vector<double>& res_work) {
    if (rhs_work.size() < (size_t)N) rhs_work.resize(N);
    if (res_work.size() < (size_t)N) res_work.resize(N);
    std::vector<double> x(N);
    std::vector<double> x_tmp(N);
    if (x0) {
        std::memcpy(x.data(), x0, (size_t)N * sizeof(double));
    } else {
        pdip_cholesky_solve(N, B_chol, c, x.data(), res_work.data());
    }
    matvec_A_block(block, x.data(), res_work.data());
    double r_pk = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:r_pk) schedule(static)
#endif
    for (int i = 0; i < N; ++i) r_pk += (c[i] - res_work[i]) * (c[i] - res_work[i]);
    r_pk = std::sqrt(r_pk) + 1e-20;
    for (int k = 0; k < fp_max_iter; ++k) {
        matvec_A_block(block, x.data(), res_work.data());
        for (int i = 0; i < N; ++i) rhs_work[i] = c[i] - res_work[i];
        matvec_B_block(block, x.data(), res_work.data());
        for (int i = 0; i < N; ++i) rhs_work[i] += res_work[i];
        pdip_cholesky_solve(N, B_chol, rhs_work.data(), x_tmp.data(), res_work.data());
        matvec_A_block(block, x_tmp.data(), res_work.data());
        double res_norm = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:res_norm) schedule(static)
#endif
        for (int i = 0; i < N; ++i) {
            double r = c[i] - res_work[i];
            res_norm += r * r;
        }
        res_norm = std::sqrt(res_norm);
        if (res_norm < exit_scale * r_pk) {
            std::memcpy(x_out, x_tmp.data(), (size_t)N * sizeof(double));
            return;
        }
        r_pk = res_norm;
        std::memcpy(x.data(), x_tmp.data(), (size_t)N * sizeof(double));
    }
    std::memcpy(x_out, x_tmp.data(), (size_t)N * sizeof(double));
}

void pdip_fp_internal(
    PDIPResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const PDIPSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
) {
    (void)verbose;
    (void)cout;
#ifdef _OPENMP
    {
        const char* env = std::getenv("OMP_NUM_THREADS");
        int nthreads = omp_get_max_threads();
        if (env && *env) {
            int v = 0;
            for (; *env >= '0' && *env <= '9'; ++env) v = v * 10 + (*env - '0');
            if (v > 0) nthreads = v;
        }
        omp_set_num_threads(nthreads);
    }
#endif
    // ===== 阶段0：参数与容器初始化 =====
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    const int n_vars = n * m, n_constraints = m + n - 1, N_block = n_constraints;
    double reg_val = (reg > 0) ? reg : 1e-6;
    double barrier = reg_val;
    double fixed_threshold = opts.fixed_threshold;
    int fp_max_iter = opts.fp_max_iter;
    double exit_scale = opts.fp_exit_scale;
    if (exit_scale <= 0) exit_scale = 1e-2;

    PdipFpProfile prof{};
    auto t_start = Clock::now();
    auto elapsed = [](Clock::time_point from) {
        return std::chrono::duration<double>(Clock::now() - from).count();
    };

    std::vector<double> cost(n_vars), eq_vector(n_constraints);
    std::memcpy(cost.data(), M.data(), sizeof(double) * static_cast<size_t>(n_vars));
    for (int i = 0; i < m; ++i) eq_vector[i] = b(i);
    for (int i = 0; i < n - 1; ++i) eq_vector[m + i] = a(i);

    std::vector<double> scale(n_vars), b1_scaled(n_vars);
    std::vector<double> x(n_vars), s(n_vars), lambda_val(n_constraints);
    std::vector<double> delta_x(n_vars), delta_lambda(n_constraints), delta_s(n_vars);
    std::vector<double> delta_lambda_final(n_constraints);
    std::vector<double> b1(n_vars), b2(n_constraints), c(n_constraints);
    std::vector<double> r_dual(n_vars), r_pri(n_constraints);
    std::vector<double> rhs_fp(n_constraints), res_fp(n_constraints);

    // ===== 阶段1：构造初始可行点 =====
    std::vector<double> initial_x(n_vars);
    solve_AAT(eq_vector.data(), n, m, lambda_val.data());
    eq_matvec_trans(lambda_val.data(), n, m, initial_x.data());
    std::vector<double> rhs_init(n_constraints);
    eq_matvec(nullptr, initial_x.data(), n, m, rhs_init.data());
    eq_matvec(nullptr, cost.data(), n, m, b1.data());
    for (int i = 0; i < n_constraints; ++i) rhs_init[i] = -barrier * rhs_init[i] - b1[i];
    solve_AAT(rhs_init.data(), n, m, lambda_val.data());
    eq_matvec_trans(lambda_val.data(), n, m, b1.data());
    std::vector<double> initial_s(n_vars);
    for (int i = 0; i < n_vars; ++i) initial_s[i] = barrier * initial_x[i] + cost[i] + b1[i];
    double dx_shift = 0, ds_shift = 0;
    for (int i = 0; i < n_vars; ++i) {
        if (initial_x[i] < 0) dx_shift = std::max(dx_shift, -1.5 * initial_x[i]);
        if (initial_s[i] < 0) ds_shift = std::max(ds_shift, -1.5 * initial_s[i]);
    }
    for (int i = 0; i < n_vars; ++i) { x[i] = initial_x[i] + dx_shift; s[i] = initial_s[i] + ds_shift; }
    double xs = 0, ss = 0, xx = 0;
    for (int i = 0; i < n_vars; ++i) xs += x[i] * s[i];
    for (int i = 0; i < n_vars; ++i) ss += s[i];
    for (int i = 0; i < n_vars; ++i) xx += x[i];
    for (int i = 0; i < n_vars; ++i) x[i] += 0.5 * xs / ss;
    xs = 0;
    for (int i = 0; i < n_vars; ++i) xs += x[i] * s[i];
    for (int i = 0; i < n_vars; ++i) xx += x[i];
    for (int i = 0; i < n_vars; ++i) s[i] += 0.5 * xs / xx;
    for (int i = 0; i < n_vars; ++i) { x[i] = std::max(x[i], 1e-10); s[i] = std::max(s[i], 1e-10); }

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
    std::vector<double> r_c_cor(n_vars), delta_s_final(n_vars), delta_x_final(n_vars);

    // ===== 阶段2：外层迭代 =====
    for (int iteration = 0; iteration < max_iter; ++iteration) {
        // 2.1 残差与缩放量
        auto t_loop = Clock::now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            scale[i] = x[i] / (barrier * x[i] + s[i] + SMALL);
        eq_matvec_trans(lambda_val.data(), n, m, b1.data());
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            r_dual[i] = barrier * x[i] + cost[i] + b1[i] - s[i];
        eq_matvec(nullptr, x.data(), n, m, r_pri.data());
        prof.eq_matvec_sec += elapsed(t_loop);
        for (int j = 0; j < m; ++j) r_pri[j] -= eq_vector[j];
        for (int i = 0; i < n - 1; ++i) r_pri[m + i] -= eq_vector[m + i];

        // 2.2 构建牛顿系统矩阵（根据稀疏度选择 B1/B2 路径）
        bool use_fp_this_iter = false;
        t_loop = Clock::now();
        if (iteration < 3) {
            build_B1_full(m, n, scale.data(), B_dense);
        } else {
            double sparsity = compute_b12_sparsity_after_threshold(m, n, scale.data(), fixed_threshold);
            const double fp_sparsity_threshold = 0.99;
            use_fp_this_iter = (fixed_threshold > 0.0) && (sparsity >= fp_sparsity_threshold);
            if (use_fp_this_iter) {
                build_B2_fixed(m, n, scale.data(), fixed_threshold, B_dense, block_b2);
            } else {
                build_B1_full(m, n, scale.data(), B_dense);
            }
        }
        prof.build_B_sec += elapsed(t_loop);

        // 2.3 Cholesky 分解（带回退）
        t_loop = Clock::now();
        const std::vector<double>& B_chol = factorize_with_fallback(
            B_dense, B_backup, B_chol_work, N_block
        );
        prof.cholesky_factor_sec += elapsed(t_loop);

        // 2.4 predictor 方向
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) b1[i] = -s[i] - r_dual[i];
        for (int i = 0; i < n_constraints; ++i) b2[i] = -r_pri[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) b1_scaled[i] = scale[i] * b1[i];
        t_loop = Clock::now();
        eq_matvec(nullptr, b1_scaled.data(), n, m, c.data());
        prof.eq_matvec_sec += elapsed(t_loop);
        for (int i = 0; i < n_constraints; ++i) c[i] -= b2[i];

        const double* x0_fp = delta_lambda_prev.empty() ? nullptr : delta_lambda_prev.data();
        if (!use_fp_this_iter) {
            t_loop = Clock::now();
            pdip_cholesky_solve(N_block, B_chol, c.data(), delta_lambda.data(), res_fp.data());
            prof.cholesky_solve_sec += elapsed(t_loop);
            prof.cholesky_solve_calls += 1;
        } else {
            t_loop = Clock::now();
            fixed_point_solve_block(N_block, block_b2, B_chol, c.data(), delta_lambda.data(), x0_fp,
                                    exit_scale, fp_max_iter, rhs_fp, res_fp);
            prof.fixed_point_sec += elapsed(t_loop);
            prof.fixed_point_calls += 1;
        }

        // 2.5 校正步（Mehrotra 风格）
        t_loop = Clock::now();
        eq_matvec_trans(delta_lambda.data(), n, m, b1.data());
        prof.eq_matvec_sec += elapsed(t_loop);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            delta_s[i] = (1 - barrier * scale[i]) * (b1[i] + r_dual[i] - barrier * x[i]);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            delta_x[i] = (-x[i] * s[i] - x[i] * delta_s[i]) / (s[i] + SMALL);

        double alpha1 = std::min(step_size(n_vars, x.data(), delta_x.data()), 1.0);
        alpha1 = std::min(alpha1, step_size(n_vars, s.data(), delta_s.data()));
        alpha1 = std::min(alpha1, 1.0);
        double mu = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:mu) schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) mu += x[i] * s[i];
        mu /= n_vars;
        double mu_aff = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:mu_aff) schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) mu_aff += (x[i] + alpha1 * delta_x[i]) * (s[i] + alpha1 * delta_s[i]);
        mu_aff /= n_vars;
        double sigma = std::min(std::pow(mu_aff / mu, 3.0), 1.0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            r_c_cor[i] = x[i] * s[i] + delta_x[i] * delta_s[i] - sigma * mu;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) b1[i] = -r_c_cor[i] / (x[i] + SMALL) - r_dual[i];
        for (int i = 0; i < n_constraints; ++i) b2[i] = -r_pri[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) b1_scaled[i] = scale[i] * b1[i];
        t_loop = Clock::now();
        eq_matvec(nullptr, b1_scaled.data(), n, m, c.data());
        prof.eq_matvec_sec += elapsed(t_loop);
        for (int i = 0; i < n_constraints; ++i) c[i] -= b2[i];
        if (!use_fp_this_iter) {
            t_loop = Clock::now();
            pdip_cholesky_solve(N_block, B_chol, c.data(), delta_lambda_final.data(), res_fp.data());
            prof.cholesky_solve_sec += elapsed(t_loop);
            prof.cholesky_solve_calls += 1;
        } else {
            t_loop = Clock::now();
            fixed_point_solve_block(N_block, block_b2, B_chol, c.data(), delta_lambda_final.data(), delta_lambda.data(),
                                    exit_scale, fp_max_iter, rhs_fp, res_fp);
            prof.fixed_point_sec += elapsed(t_loop);
            prof.fixed_point_calls += 1;
        }

        t_loop = Clock::now();
        eq_matvec_trans(delta_lambda_final.data(), n, m, b1.data());
        prof.eq_matvec_sec += elapsed(t_loop);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            delta_s_final[i] = (1 - barrier * scale[i]) * (b1[i] + r_dual[i] - barrier * r_c_cor[i] / (s[i] + SMALL));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i)
            delta_x_final[i] = (-r_c_cor[i] - x[i] * delta_s_final[i]) / (s[i] + SMALL);
        double alpha2 = std::min(step_size(n_vars, x.data(), delta_x_final.data()), step_size(n_vars, s.data(), delta_s_final.data()));
        alpha2 = std::min(0.99 * alpha2, 1.0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) {
            x[i] += alpha2 * delta_x_final[i];
            s[i] += alpha2 * delta_s_final[i];
        }
        for (int i = 0; i < n_constraints; ++i) lambda_val[i] += alpha2 * delta_lambda_final[i];
        delta_lambda_prev = delta_lambda_final;

        // 记录 marginal error（历史始终写入；是否参与停机由 fp_stop_gap_mu_only 决定）
        const double current_mar_err = marginal_error(n, m, x.data(), eq_vector.data());
        mar_err_history.push_back(current_mar_err);
        double obj = 0;
        for (int i = 0; i < n_vars; ++i) obj += cost[i] * x[i];
        for (int i = 0; i < n_vars; ++i) obj += (reg_val / 2.0) * x[i] * x[i];
        obj_vals.push_back(obj);
        run_times.push_back(std::chrono::duration<double>(Clock::now() - t_start).count() * 1000.0);

        // 2.6 收敛判定
        t_loop = Clock::now();
        eq_matvec_trans(lambda_val.data(), n, m, b1.data());
        for (int i = 0; i < n_vars; ++i) r_dual[i] = barrier * x[i] + cost[i] + b1[i] - s[i];
        eq_matvec(nullptr, x.data(), n, m, r_pri.data());
        prof.eq_matvec_sec += elapsed(t_loop);
        for (int j = 0; j < m; ++j) r_pri[j] -= eq_vector[j];
        for (int i = 0; i < n - 1; ++i) r_pri[m + i] -= eq_vector[m + i];
        double dual_gap = 0, primal_gap = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:dual_gap) schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) dual_gap += r_dual[i] * r_dual[i];
        dual_gap = std::sqrt(dual_gap);
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:primal_gap) schedule(static)
#endif
        for (int i = 0; i < n_constraints; ++i) primal_gap += r_pri[i] * r_pri[i];
        primal_gap = std::sqrt(primal_gap);
        double norm_eq = 0;
        for (int i = 0; i < n_constraints; ++i) norm_eq += eq_vector[i] * eq_vector[i];
        norm_eq = std::sqrt(norm_eq);
        double denom_pri = 1.0 + norm_eq;
        double norm_cost = 0, norm_at_lambda = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:norm_cost,norm_at_lambda) schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) {
            norm_cost += cost[i] * cost[i];
            norm_at_lambda += b1[i] * b1[i];
        }
        double denom_dual = 1.0 + std::sqrt(norm_cost) + std::sqrt(norm_at_lambda);
        dual_gap /= denom_dual;
        primal_gap /= denom_pri;
        mu = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:mu) schedule(static)
#endif
        for (int i = 0; i < n_vars; ++i) mu += x[i] * s[i];
        mu /= n_vars;
        // 收敛：默认仅 (rp∧rd∧μ)；fp_stop_gap_mu_only==false 时允许 mar 捷径
        const bool by_gap_mu =
            (primal_gap < tol && dual_gap < tol && mu < tol);
        const bool by_mar = (current_mar_err < tol);
        const bool stop = opts.fp_stop_gap_mu_only ? by_gap_mu : (by_gap_mu || by_mar);
        if (stop) {
            result.converged = true;
            result.niter = iteration + 1;
            break;
        }
        result.niter = iteration + 1;
    }

    result.plan.resize(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            result.plan(i, j) = x[i * m + j];
    result.obj_vals = std::move(obj_vals);
    result.mar_errs = std::move(mar_err_history);
    result.run_times = std::move(run_times);
    result.t_build_B = prof.build_B_sec;
    result.t_chol_factor = prof.cholesky_factor_sec;
    result.t_chol_solve = prof.cholesky_solve_sec;
    result.t_eq_matvec = prof.eq_matvec_sec;
    double t_total = std::chrono::duration<double>(Clock::now() - t_start).count();
    result.t_other = std::max(0.0, t_total - (result.t_build_B + result.t_chol_factor + result.t_chol_solve + result.t_eq_matvec));
}

}  // namespace PDIP
