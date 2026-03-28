/**
 * Role of this file:
 * 1) PDIP-CG kernel (same lineage as the first version);
 * 2) Unified regot primary fields;
 * 3) Optional per-phase timing output for profiling.
 */
#define NOMINMAX
#include "pdip_solvers.h"
#include "pdip_transport_ops.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace Eigen;

namespace PDIP {

using Vector = VectorXd;
using RowVector = RowVectorXd;
using MatRowMajor = transport::MatRowMajor;

struct PdipTiming {
    double build_B = 0, B_compute = 0, build_A = 0, pcg1 = 0, pcg2 = 0, other = 0;
    void reset() { build_B = B_compute = build_A = pcg1 = pcg2 = other = 0; }
};
static PdipTiming s_timing;
static bool s_timing_enabled() {
    const char* v = std::getenv("PDIP_CG_TIMING");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}
static bool use_eigen_cg() {
    const char* v = std::getenv("PDIP_USE_EIGEN_CG");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static const double SMALL = 1e-50;
static const int STABILIZATION_WINDOW = 30;
static const double STABILIZATION_MAX_RANGE = 0.1;
static const int STABILIZATION_MIN_ITER = 30;

static void A_matvec(int n, int m, const Vector& x, Vector& y) {
    y.setZero(m + n - 1);
    transport::A_matvec_from_x(n, m, x.data(), nullptr, y.data());
}

static void A_matvec_scaled(int n, int m, const Vector& scale, const Vector& x, Vector& y) {
    y.setZero(m + n - 1);
    transport::A_matvec_from_x(n, m, x.data(), scale.data(), y.data());
}

static void AT_matvec(int n, int m, const Vector& lambda, Vector& y) {
    transport::AT_matvec(n, m, lambda.data(), y.data());
}

static void solve_AAT(const Vector& rhs_in, int n_supp, int m_dem, Vector& out) {
    transport::solve_AAT_vec(rhs_in, n_supp, m_dem, out);
}

static double step_size(const Vector& x, const Vector& dx) {
    ArrayXd cand = (dx.array() < 0).select(-x.array() / (dx.array() + SMALL), 1.0);
    return (std::min)(1.0, cand.minCoeff());
}

static double compute_mar_err(int n, int m, const Vector& x, const Vector& eq_vector,
                              Vector* row_sums, Vector* col_sums, Vector* a_marg) {
    Map<const MatRowMajor> X(x.data(), n, m);
    *row_sums = X.rowwise().sum();
    *col_sums = X.colwise().sum().transpose();
    if (a_marg->size() != static_cast<Eigen::Index>(n)) a_marg->resize(n);
    a_marg->head(n - 1) = eq_vector.tail(n - 1);
    (*a_marg)(n - 1) = 1.0 - eq_vector.tail(n - 1).sum();
    double err1 = (*row_sums - *a_marg).norm();
    double err2 = (*col_sums - eq_vector.head(m)).norm();
    return (std::max)(err1, err2);
}

static bool mar_err_stabilized(const std::vector<double>& mar_err_list) {
    int len = static_cast<int>(mar_err_list.size());
    if (len < STABILIZATION_MIN_ITER) return false;
    int start = (std::max)(0, len - STABILIZATION_WINDOW);
    double max_log = -1e30, min_log = 1e30;
    for (int i = start; i < len; ++i) {
        double v = std::log10((std::max)(mar_err_list[i], 1e-20));
        max_log = (std::max)(max_log, v);
        min_log = (std::min)(min_log, v);
    }
    return (max_log - min_log) <= STABILIZATION_MAX_RANGE;
}

static double dynamic_threshold(const Vector& D_B12, int iter_cur, int n_supp, int M_dem, double reg_val) {
    std::vector<double> pos;
    for (Eigen::Index ii = 0; ii < D_B12.size(); ++ii)
        if (D_B12(ii) > 0) pos.push_back(D_B12(ii));
    if (pos.empty()) return 1e-9;
    double scale_ref = (reg_val > 0) ? (1.0 / reg_val) : 0;
    if (scale_ref < 1e-16) {
        size_t mid = pos.size() / 2;
        std::nth_element(pos.begin(), pos.begin() + mid, pos.end());
        scale_ref = pos[mid];
    }
    scale_ref = (std::max)(scale_ref, 1e-16);

    double keep_ratio;
    bool use_new_sparsity = false;
    double keep_factor = 3.0;
    const char* env_keep = std::getenv("PDIP_SPARSITY_KEEP");
    if (env_keep) keep_factor = std::atof(env_keep);
    if (keep_factor <= 1.0 || keep_factor > 10.0) {
        keep_ratio = 1.0 / (std::max)(n_supp, 1);
    } else {
        use_new_sparsity = true;
        const int k0 = 15;
        double progress = (std::min)(static_cast<double>(iter_cur) / (std::max)(k0, 1), 1.0);
        double eff_keep_factor = 1.0 + (keep_factor - 1.0) * progress;
        keep_ratio = (std::min)(1.0, eff_keep_factor / (std::max)(n_supp, 1));
    }
    size_t idx = static_cast<size_t>((1.0 - keep_ratio) * pos.size());
    if (idx >= pos.size()) idx = 0;
    std::nth_element(pos.begin(), pos.begin() + idx, pos.end());
    double threshold_pct = pos[idx];

    int iter_val = (std::min)((std::max)(iter_cur, 0), 52);
    double t_fac = std::pow(1.48, iter_val);
    double blend = use_new_sparsity
        ? (std::min)(1.0, (iter_cur + 1) / 20.0)
        : (std::min)(1.0, static_cast<double>(iter_cur) / 42.0);
    double th = (scale_ref / (std::max)(n_supp, 1)) * t_fac * (1.0 - 0.52 * blend) + threshold_pct * 0.52 * blend;
    if (use_new_sparsity) {
        th = (std::min)(th, threshold_pct);
        double min_th = (std::max)(1e-9, (std::min)(scale_ref * 1e-6, threshold_pct));
        th = (std::max)(th, min_th);
    } else {
        th = (std::max)(th, (std::max)(1e-9, scale_ref * 1e-6));
    }
    th = (std::min)(th, scale_ref * 0.5);
    return th;
}

static void A_block_matvec(int M_dem, int n12, const Vector& B11_diag, const Vector& B22_diag,
                           const Vector& D_B12, const Vector& x, Vector& y) {
    y.resize(M_dem + n12);
    Map<const Matrix<double, Dynamic, Dynamic, ColMajor>> D(D_B12.data(), M_dem, n12);
    y.head(M_dem).noalias() = B11_diag.cwiseProduct(x.head(M_dem)) + D * x.tail(n12);
    y.tail(n12).noalias() = B22_diag.cwiseProduct(x.tail(n12)) + D.transpose() * x.head(M_dem);
}

static void build_full_A_sparse(int M_dem, int n12,
                                const Vector& B11_diag, const Vector& B22_diag, const Vector& D_B12,
                                SparseMatrix<double>& A_out) {
    const int N = M_dem + n12;
    const size_t max_trips = static_cast<size_t>(N + 2 * M_dem * n12);
    std::vector<Triplet<double>> trips;
    trips.reserve(max_trips);
    for (int i = 0; i < M_dem; ++i) trips.emplace_back(i, i, B11_diag(i));
    for (int i = 0; i < n12; ++i) trips.emplace_back(M_dem + i, M_dem + i, B22_diag(i));
    for (int i = 0; i < M_dem; ++i)
        for (int j = 0; j < n12; ++j) {
            double v = D_B12(i + j * M_dem);
            trips.emplace_back(i, M_dem + j, v);
            trips.emplace_back(M_dem + j, i, v);
        }
    A_out.resize(N, N);
    A_out.setFromTriplets(trips.begin(), trips.end());
}

struct BPreconditioner {
    SimplicialLDLT<SparseMatrix<double>>* solver = nullptr;
    void compute(const SparseMatrix<double>&) {}
    Vector solve(const Vector& b) const { return solver ? solver->solve(b) : b; }
    ComputationInfo info() const { return solver ? solver->info() : Success; }
};

static void build_B_and_block_A(int M_dem, int n_supp, const Vector& scale, int iter_cur, double reg_val,
                                SparseMatrix<double>& B, Vector& B11_diag, Vector& B22_diag, Vector& D_B12) {
    const int n12 = n_supp - 1;
    const int N = M_dem + n12;
    B11_diag.resize(M_dem);
    B22_diag.resize(n12);
    D_B12.resize(M_dem * n12);
    Map<const Matrix<double, Dynamic, Dynamic, ColMajor>> S(scale.data(), M_dem, n_supp);
    B11_diag = S.rowwise().sum();
    B22_diag = S.leftCols(n12).colwise().sum().transpose();
    D_B12 = Map<const Vector>(scale.data(), M_dem * n12);
    double th = dynamic_threshold(D_B12, iter_cur, n_supp, M_dem, reg_val);
    const size_t max_trips = static_cast<size_t>(N + 2 * M_dem * n12);
    std::vector<Triplet<double>> B_trips;
    B_trips.reserve(max_trips);
    for (int i = 0; i < M_dem; ++i) B_trips.emplace_back(i, i, B11_diag(i));
    for (int i = 0; i < n12; ++i) B_trips.emplace_back(M_dem + i, M_dem + i, B22_diag(i));
    for (int i = 0; i < M_dem; ++i)
        for (int j = 0; j < n12; ++j) {
            double v = D_B12(i + j * M_dem);
            if (v >= th) {
                B_trips.emplace_back(i, M_dem + j, v);
                B_trips.emplace_back(M_dem + j, i, v);
            }
        }
    B.resize(N, N);
    B.setFromTriplets(B_trips.begin(), B_trips.end());
}

template<typename PrecondType>
int pcg_solve_matrix_free(int M_dem, int n12,
                          const Vector& B11_diag, const Vector& B22_diag, const Vector& D_B12,
                          const Vector& c_vec, PrecondType& B_solver,
                          Vector& x_out, const Vector* x0, double rtol, double atol, int max_iter,
                          Vector& r, Vector& z, Vector& p, Vector& Ap) {
    const int N = M_dem + n12;
    r.resize(N); z.resize(N); p.resize(N); Ap.resize(N);
    A_block_matvec(M_dem, n12, B11_diag, B22_diag, D_B12, x_out, Ap);
    r.noalias() = c_vec - Ap;
    z = B_solver.solve(r);
    p = z;
    double rz = r.dot(z);
    double b_norm = c_vec.norm();
    double tol = (std::max)(rtol * b_norm, atol);
    for (int it = 0; it < max_iter; ++it) {
        A_block_matvec(M_dem, n12, B11_diag, B22_diag, D_B12, p, Ap);
        double pAp = p.dot(Ap);
        if (pAp <= 0) return -1;
        double alpha = rz / pAp;
        x_out.noalias() += alpha * p;
        r.noalias() -= alpha * Ap;
        if (r.norm() <= tol) return it + 1;
        z = B_solver.solve(r);
        double rz_new = r.dot(z);
        double beta = rz_new / rz;
        rz = rz_new;
        p.noalias() = z + beta * p;
    }
    return -1;
}

void pdip_cg_internal(
    PDIPResult& result,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    const PDIPSolverOpts& opts,
    double tol, int max_iter, int verbose, std::ostream& cout
) {
    (void)verbose;
    (void)cout;
    // Dimensions
    const int n = static_cast<int>(a.size()), m = static_cast<int>(b.size());
    const int n_vars = n * m, n_constraints = m + n - 1;
    double reg_val = (reg > 0) ? reg : 1e-6;
    double barrier = reg_val;
    int cg_max_iter = opts.cg_max_iter;

    Map<const Vector> cost_vec(M.data(), n_vars);
    Vector eq_vec(n_constraints);
    eq_vec.head(m) = b;
    eq_vec.tail(n - 1) = a.head(n - 1);

    auto t_start = std::chrono::steady_clock::now();
    Vector x(n_vars), s(n_vars), lambda_val(n_constraints);
    Vector scale(n_vars), b1_scaled(n_vars), delta_x(n_vars), delta_lambda(n_constraints), delta_s(n_vars);
    Vector delta_lambda_final(n_constraints), b1(n_vars), b2(n_constraints), c_vec(n_constraints);
    Vector r_dual(n_vars), r_pri(n_constraints), at_lambda(n_vars);

    // Phase 1: initial point
    solve_AAT(eq_vec, n, m, lambda_val);
    AT_matvec(n, m, lambda_val, b1);
    Vector initial_x = b1;
    b2.setZero();
    A_matvec(n, m, initial_x, b2);
    b2 = -barrier * b2;
    A_matvec(n, m, cost_vec, r_pri);
    b2 -= r_pri;
    solve_AAT(b2, n, m, lambda_val);
    AT_matvec(n, m, lambda_val, at_lambda);
    Vector initial_s = barrier * initial_x + cost_vec + at_lambda;
    double dx_shift = 0.0, ds_shift = 0.0;
    for (int ii = 0; ii < n_vars; ++ii) {
        if (initial_x(ii) < 0) dx_shift = (std::max)(dx_shift, -1.5 * initial_x(ii));
        if (initial_s(ii) < 0) ds_shift = (std::max)(ds_shift, -1.5 * initial_s(ii));
    }
    x = initial_x.array() + dx_shift;
    s = initial_s.array() + ds_shift;
    double xs = x.dot(s), ss = s.sum(), xx = x.sum();
    x.array() += 0.5 * xs / (ss + SMALL);
    xs = x.dot(s);
    xx = x.sum();
    s.array() += 0.5 * xs / (xx + SMALL);
    x = x.cwiseMax(1e-10);
    s = s.cwiseMax(1e-10);

    Vector delta_lambda_prev;
    std::vector<double> mar_err_history, time_sec_history, obj_history;
    const int N_block = m + n - 1, n12 = n - 1;
    Vector B11_diag(m), B22_diag(n12), D_B12(m * n12);
    Vector pcg_r(n_constraints), pcg_z(n_constraints), pcg_p(n_constraints), pcg_Ap(n_constraints);
    Vector mar_row_sums(n), mar_col_sums(m), mar_a_marg(n), r_c_cor(n_vars), delta_s_final(n_vars), delta_x_final(n_vars);
    const bool do_timing = s_timing_enabled();
    if (do_timing) s_timing.reset();
    double total_loop_sec = 0;

    // Phase 2: outer iterations
    for (int iteration = 0; iteration < max_iter; ++iteration) {
        // 2.1 Residuals and scaling
        auto t_iter_start = std::chrono::steady_clock::now();
        scale = x.array() / (barrier * x.array() + s.array() + SMALL);
        AT_matvec(n, m, lambda_val, at_lambda);
        r_dual = barrier * x + cost_vec + at_lambda - s;
        A_matvec(n, m, x, r_pri);
        r_pri.head(m) -= eq_vec.head(m);
        r_pri.tail(n - 1) -= eq_vec.tail(n - 1);

        // 2.2 Build preconditioner matrix B
        SparseMatrix<double> B_sp;
        auto t0 = std::chrono::steady_clock::now();
        build_B_and_block_A(m, n, scale, iteration, reg_val, B_sp, B11_diag, B22_diag, D_B12);
        if (do_timing) s_timing.build_B += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        SimplicialLDLT<SparseMatrix<double>> B_solver;
        t0 = std::chrono::steady_clock::now();
        for (int shift_try = 0; shift_try < 4; ++shift_try) {
            double shift = (shift_try == 0) ? 0 : (shift_try == 1 ? 1e-12 : (shift_try == 2 ? 1e-10 : 1e-8));
            SparseMatrix<double> B_shift = B_sp;
            if (shift > 0)
                for (int i = 0; i < N_block; ++i) B_shift.coeffRef(i, i) += shift;
            B_solver.compute(B_shift);
            if (B_solver.info() == Success) break;
        }
        if (do_timing) s_timing.B_compute += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        double r_p_norm = r_pri.norm();
        double cg_rel_early = 0.1, cg_rel_late = 0.01;
        int cg_decay = 40;
        double t_blend = (std::min)(static_cast<double>(iteration), static_cast<double>(cg_decay)) / (std::max)(cg_decay, 1);
        double rel_tol = cg_rel_early * std::pow(cg_rel_late / cg_rel_early, t_blend);
        double cg_tol = (std::max)(rel_tol * r_p_norm, 1e-12);

        // 2.3 Predictor direction (first linear system)
        b1 = -s - r_dual;
        b2 = -r_pri;
        b1_scaled = scale.array() * b1.array();
        A_matvec_scaled(n, m, scale, b1, c_vec);
        c_vec -= b2;
        delta_lambda.setZero();
        const Vector* x0_ptr = (delta_lambda_prev.size() == n_constraints) ? &delta_lambda_prev : nullptr;
        if (x0_ptr) delta_lambda = *x0_ptr;
        double c_norm = c_vec.norm() + 1e-50;
        double rtol_cg = (std::min)(cg_tol / c_norm, 0.1);
        t0 = std::chrono::steady_clock::now();
        if (use_eigen_cg()) {
            SparseMatrix<double> A_sparse;
            auto t_build_A = std::chrono::steady_clock::now();
            build_full_A_sparse(m, n12, B11_diag, B22_diag, D_B12, A_sparse);
            double build_A_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build_A).count();
            if (do_timing) s_timing.build_A += build_A_sec;
            ConjugateGradient<SparseMatrix<double>, Lower|Upper, BPreconditioner> cg;
            cg.compute(A_sparse);
            cg.preconditioner().solver = &B_solver;
            cg.setTolerance(cg_tol / c_norm);
            cg.setMaxIterations(cg_max_iter);
            delta_lambda = cg.solveWithGuess(c_vec, delta_lambda);
        } else {
            pcg_solve_matrix_free(m, n12, B11_diag, B22_diag, D_B12, c_vec, B_solver,
                                  delta_lambda, x0_ptr, rtol_cg, cg_tol, cg_max_iter,
                                  pcg_r, pcg_z, pcg_p, pcg_Ap);
        }
        if (do_timing) s_timing.pcg1 += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // 2.4 Corrector step (second linear system)
        AT_matvec(n, m, delta_lambda, b1);
        delta_s = ((1.0 - barrier * scale.array()) * (b1 + r_dual - barrier * x).array()).matrix();
        delta_x = ((-x.array() * s.array() - x.array() * delta_s.array()) / (s.array() + SMALL)).matrix();

        double alpha1 = (std::min)(step_size(x, delta_x), 1.0);
        alpha1 = (std::min)(alpha1, step_size(s, delta_s));
        alpha1 = (std::min)(alpha1, 1.0);
        double mu = x.dot(s) / n_vars;
        double mu_aff = (x + alpha1 * delta_x).dot(s + alpha1 * delta_s) / n_vars;
        double sigma = (std::min)(std::pow(mu_aff / mu, 3.0), 1.0);
        r_c_cor = (x.array() * s.array() + delta_x.array() * delta_s.array() - sigma * mu).matrix();
        b1 = ((-r_c_cor.array() / (x.array() + SMALL)) - r_dual.array()).matrix();
        b2 = -r_pri;
        b1_scaled = scale.array() * b1.array();
        A_matvec_scaled(n, m, scale, b1, c_vec);
        c_vec -= b2;
        delta_lambda_final = delta_lambda;
        c_norm = c_vec.norm() + 1e-50;
        rtol_cg = (std::min)(cg_tol / c_norm, 0.1);
        t0 = std::chrono::steady_clock::now();
        if (use_eigen_cg()) {
            SparseMatrix<double> A_sparse;
            auto t_build_A = std::chrono::steady_clock::now();
            build_full_A_sparse(m, n12, B11_diag, B22_diag, D_B12, A_sparse);
            double build_A_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build_A).count();
            if (do_timing) s_timing.build_A += build_A_sec;
            ConjugateGradient<SparseMatrix<double>, Lower|Upper, BPreconditioner> cg;
            cg.compute(A_sparse);
            cg.preconditioner().solver = &B_solver;
            cg.setTolerance(cg_tol / c_norm);
            cg.setMaxIterations(cg_max_iter);
            delta_lambda_final = cg.solveWithGuess(c_vec, delta_lambda);
        } else {
            pcg_solve_matrix_free(m, n12, B11_diag, B22_diag, D_B12, c_vec, B_solver,
                                  delta_lambda_final, &delta_lambda, rtol_cg, cg_tol, cg_max_iter,
                                  pcg_r, pcg_z, pcg_p, pcg_Ap);
        }
        if (do_timing) s_timing.pcg2 += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        AT_matvec(n, m, delta_lambda_final, b1);
        delta_s_final = ((1.0 - barrier * scale.array()) * (b1.array() + r_dual.array() - barrier * r_c_cor.array() / (s.array() + SMALL))).matrix();
        delta_x_final = ((-r_c_cor.array() - x.array() * delta_s_final.array()) / (s.array() + SMALL)).matrix();
        double alpha2 = (std::min)(step_size(x, delta_x_final), step_size(s, delta_s_final));
        alpha2 = (std::min)(0.99 * alpha2, 1.0);
        x.noalias() += alpha2 * delta_x_final;
        lambda_val.noalias() += alpha2 * delta_lambda_final;
        s.noalias() += alpha2 * delta_s_final;
        delta_lambda_prev.resize(n_constraints);
        delta_lambda_prev = delta_lambda_final;

        // 2.5 Convergence check and history
        r_dual = barrier * x + cost_vec + at_lambda - s;
        A_matvec(n, m, x, r_pri);
        r_pri.head(m) -= eq_vec.head(m);
        r_pri.tail(n - 1) -= eq_vec.tail(n - 1);
        AT_matvec(n, m, lambda_val, at_lambda);
        double primal_gap = r_pri.norm() / (1.0 + eq_vec.norm());
        double dual_gap = r_dual.norm() / (1.0 + cost_vec.norm() + at_lambda.norm());
        mu = x.dot(s) / n_vars;

        double current_mar_err = compute_mar_err(n, m, x, eq_vec, &mar_row_sums, &mar_col_sums, &mar_a_marg);
        if (!std::isfinite(current_mar_err) || !std::isfinite(primal_gap) || !std::isfinite(dual_gap) || !std::isfinite(mu)) {
            // Numerics unstable: exit so the caller sees converged=false.
            break;
        }
        // Early stop if marginal error blows up to avoid NaN plans and wasted wall time
        if (current_mar_err > 1e3 || !x.allFinite() || !s.allFinite()) {
            result.niter = iteration + 1;
            break;
        }
        mar_err_history.push_back(current_mar_err);
        time_sec_history.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() * 1000.0);
        obj_history.push_back(cost_vec.dot(x) + (reg_val / 2.0) * x.squaredNorm());

        if (do_timing) total_loop_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_iter_start).count();
        // gap+mu use tol; mar / stability window use opts.cg_mar_tol (default 1e-10)
        bool by_gap = (primal_gap < tol && dual_gap < tol && mu < tol);
        bool by_mar_tol = (current_mar_err < opts.cg_mar_tol);
        bool by_stable = mar_err_stabilized(mar_err_history) && (current_mar_err < opts.cg_mar_tol);
        const bool stop = opts.cg_stop_gap_mu_only
            ? by_gap
            : (by_gap || by_mar_tol || by_stable);
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
            result.plan(i, j) = x(i * m + j);
    result.obj_vals = std::move(obj_history);
    result.mar_errs = std::move(mar_err_history);
    result.run_times = std::move(time_sec_history);

    if (do_timing) {
        s_timing.other = total_loop_sec - s_timing.build_B - s_timing.B_compute - s_timing.build_A - s_timing.pcg1 - s_timing.pcg2;
        if (s_timing.other < 0) s_timing.other = 0;
        double tot = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() + 1e-20;
        std::FILE* fp = std::fopen("pdip_cg_timing.txt", "w");
        if (fp) {
            if (s_timing.build_A > 0)
                std::fprintf(fp,
                    "n=%d m=%d iters=%d total=%.3fs | build_B=%.3fs B_compute=%.3fs build_A=%.3fs pcg1=%.3fs pcg2=%.3fs other=%.3fs [Eigen CG]\n",
                    n, m, result.niter, tot,
                    s_timing.build_B, s_timing.B_compute, s_timing.build_A, s_timing.pcg1, s_timing.pcg2, s_timing.other);
            else
                std::fprintf(fp,
                    "n=%d m=%d iters=%d total=%.3fs | build_B=%.3fs(%.0f%%) B_compute=%.3fs(%.0f%%) pcg1=%.3fs(%.0f%%) pcg2=%.3fs(%.0f%%) other=%.3fs(%.0f%%)\n",
                    n, m, result.niter, tot,
                    s_timing.build_B, 100.0 * s_timing.build_B / tot,
                    s_timing.B_compute, 100.0 * s_timing.B_compute / tot,
                    s_timing.pcg1, 100.0 * s_timing.pcg1 / tot,
                    s_timing.pcg2, 100.0 * s_timing.pcg2 / tot,
                    s_timing.other, 100.0 * s_timing.other / tot);
            std::fclose(fp);
        }
    }
}

}  // namespace PDIP
