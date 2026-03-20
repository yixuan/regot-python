// 文件职责：
// 1) 将 C++ 求解器绑定到 Python；
// 2) 统一解析 kwargs 到各算法配置；
// 3) 维护 regot 对外返回对象字段一致性。
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "qrot_solvers.h"
#include "sinkhorn_solvers.h"
#include "pdip_solvers.h"

namespace py = pybind11;
using namespace pybind11::literals;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefConstVec = Eigen::Ref<const Vector>;
using RefConstMat = Eigen::Ref<const Matrix>;

using QROT::QROTResult;
using QROT::QROTSolverOpts;
using Sinkhorn::SinkhornResult;
using Sinkhorn::SinkhornSolverOpts;
using PDIP::PDIPResult;
using PDIP::PDIPSolverOpts;

// Set solver_opts from Python-side keyword arguments
inline void parse_qrot_opts(
    QROTSolverOpts &solver_opts, const py::kwargs &kwargs
)
{
    // Used in ASSN, GRSSN
    // 0 - CG
    // 1 - SimplicialLDLT
    // 2 - SimplicialLLT
    // 3 - SparseLU
    if (kwargs.contains("method"))
    {
    	solver_opts.method = py::int_(kwargs["method"]);
    }
    // Used in GRSSN
    if (kwargs.contains("shift"))
    {
        solver_opts.shift = py::float_(kwargs["shift"]);
    }
}

QROTResult qrot_apdagd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_apdagd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_assn(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_assn_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_bcd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_bcd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_gd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_gd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_grssn(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    solver_opts.shift = 0.001;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_grssn_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_lbfgs_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_lbfgs_dual_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_lbfgs_semi_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_lbfgs_semi_dual_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_pdaam(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_pdaam_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

/*
QROTResult qrot_s5n(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    parse_qrot_opts(solver_opts, kwargs);

    QROT::qrot_s5n_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}
*/



// Set solver_opts from Python-side keyword arguments
inline void parse_sinkhorn_opts(
    SinkhornSolverOpts &solver_opts, const py::kwargs &kwargs
)
{
    // Used in SSNS
    // 0 - CG
    // 1 - SimplicialLDLT
    // 2 - SimplicialLLT
    // 3 - SparseLU
    if (kwargs.contains("method"))
    {
    	solver_opts.method = py::int_(kwargs["method"]);
    }
    // Used in SSNS
    if (kwargs.contains("mu0"))
    {
        solver_opts.mu0 = py::float_(kwargs["mu0"]);
    }
    // Used in sparse Newton and SPLR
    if (kwargs.contains("shift"))
    {
        solver_opts.shift = py::float_(kwargs["shift"]);
    }
    if (kwargs.contains("density"))
    {
        solver_opts.density = py::float_(kwargs["density"]);
    }
}

SinkhornResult sinkhorn_apdagd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_apdagd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_bcd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_bcd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_lbfgs_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_lbfgs_dual_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_newton(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_newton_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_sparse_newton(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    // Use LDLT as default
    solver_opts.method = 1;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_sparse_newton_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_ssns(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    // Use LDLT as default
    solver_opts.method = 1;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_ssns_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_splr(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    // Use LDLT as default
    solver_opts.method = 1;
    parse_sinkhorn_opts(solver_opts, kwargs);

    Sinkhorn::sinkhorn_splr_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

inline void parse_pdip_opts(
    PDIPSolverOpts &solver_opts, const py::kwargs &kwargs
)
{
    // cg_max_iter: CG 内层迭代上限
    if (kwargs.contains("cg_max_iter"))
    {
        solver_opts.cg_max_iter = py::int_(kwargs["cg_max_iter"]);
    }
    // fixed_threshold: FP 稀疏阈值，影响 B2/固定点路径是否启用
    if (kwargs.contains("fixed_threshold"))
    {
        solver_opts.fixed_threshold = py::float_(kwargs["fixed_threshold"]);
    }
    // fp_max_iter: FP 内层固定点迭代上限
    if (kwargs.contains("fp_max_iter"))
    {
        solver_opts.fp_max_iter = py::int_(kwargs["fp_max_iter"]);
    }
    // fp_exit_scale: FP 内层停止阈值比例
    if (kwargs.contains("fp_exit_scale"))
    {
        solver_opts.fp_exit_scale = py::float_(kwargs["fp_exit_scale"]);
    }
    if (kwargs.contains("fp_stop_gap_mu_only"))
    {
        solver_opts.fp_stop_gap_mu_only = py::bool_(kwargs["fp_stop_gap_mu_only"]);
    }
    if (kwargs.contains("cg_stop_gap_mu_only"))
    {
        solver_opts.cg_stop_gap_mu_only = py::bool_(kwargs["cg_stop_gap_mu_only"]);
    }
    if (kwargs.contains("cg_mar_tol"))
    {
        solver_opts.cg_mar_tol = py::float_(kwargs["cg_mar_tol"]);
    }
}

PDIPResult pdip_cg(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    PDIPResult result;
    PDIPSolverOpts solver_opts;
    parse_pdip_opts(solver_opts, kwargs);
    // 绑定层统一：以 QROT 的坐标约定为标准，
    // 让 PDIP 的内部求解使用转置后的成本矩阵输入。
    Matrix M_T = M.transpose();
    PDIP::pdip_cg_internal(
        result, M_T, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);
    return result;
}

PDIPResult pdip_fp(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    PDIPResult result;
    PDIPSolverOpts solver_opts;
    parse_pdip_opts(solver_opts, kwargs);
    // 绑定层统一：以 QROT 的坐标约定为标准，
    // 让 PDIP 的内部求解使用转置后的成本矩阵输入。
    Matrix M_T = M.transpose();
    PDIP::pdip_fp_internal(
        result, M_T, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);
    return result;
}



PYBIND11_MODULE(_internal, m) {
    // QROT solvers
    m.def("qrot_apdagd", &qrot_apdagd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_assn", &qrot_assn,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_bcd", &qrot_bcd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_gd", &qrot_gd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_grssn", &qrot_grssn,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_lbfgs_dual", &qrot_lbfgs_dual,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_lbfgs_semi_dual", &qrot_lbfgs_semi_dual,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("qrot_pdaam", &qrot_pdaam,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    /*
    m.def("qrot_s5n", &qrot_s5n,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    */

    // Sinkhorn solvers
    m.def("sinkhorn_apdagd", &sinkhorn_apdagd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("sinkhorn_bcd", &sinkhorn_bcd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("sinkhorn_lbfgs_dual", &sinkhorn_lbfgs_dual,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("sinkhorn_newton", &sinkhorn_newton,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("sinkhorn_sparse_newton", &sinkhorn_sparse_newton,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("sinkhorn_ssns", &sinkhorn_ssns,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("sinkhorn_splr", &sinkhorn_splr,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = 0);
    // PDIP solvers
    m.def("pdip_cg", &pdip_cg,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-8, "max_iter"_a = 1000, "verbose"_a = 0);
    m.def("pdip_fp", &pdip_fp,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-8, "max_iter"_a = 1000, "verbose"_a = 0);

    // Returned object
    py::class_<QROTResult>(m, "qrot_result")
        .def(py::init<>())
        .def_readwrite("niter", &QROTResult::niter)
        .def_readwrite("dual", &QROTResult::dual)
        .def_readwrite("plan", &QROTResult::plan)
        .def_readwrite("obj_vals", &QROTResult::obj_vals)
        .def_readwrite("prim_vals", &QROTResult::prim_vals)
        .def_readwrite("mar_errs", &QROTResult::mar_errs)
        .def_readwrite("run_times", &QROTResult::run_times);

    py::class_<SinkhornResult>(m, "sinkhorn_result")
        .def(py::init<>())
        .def_readwrite("niter", &SinkhornResult::niter)
        .def_readwrite("dual", &SinkhornResult::dual)
        .def_readwrite("plan", &SinkhornResult::plan)
        .def_readwrite("obj_vals", &SinkhornResult::obj_vals)
        // .def_readwrite("prim_vals", &SinkhornResult::prim_vals)
        .def_readwrite("mar_errs", &SinkhornResult::mar_errs)
        .def_readwrite("run_times", &SinkhornResult::run_times)
        .def_readwrite("densities", &SinkhornResult::densities);
    py::class_<PDIPResult>(m, "pdip_result")
        .def(py::init<>())
        .def_readwrite("niter", &PDIPResult::niter)
        .def_readwrite("converged", &PDIPResult::converged)
        .def_readwrite("plan", &PDIPResult::plan)
        .def_readwrite("obj_vals", &PDIPResult::obj_vals)
        .def_readwrite("mar_errs", &PDIPResult::mar_errs)
        .def_readwrite("run_times", &PDIPResult::run_times)
        .def_readwrite("t_build_B", &PDIPResult::t_build_B)
        .def_readwrite("t_chol_factor", &PDIPResult::t_chol_factor)
        .def_readwrite("t_chol_solve", &PDIPResult::t_chol_solve)
        .def_readwrite("t_eq_matvec", &PDIPResult::t_eq_matvec)
        .def_readwrite("t_other", &PDIPResult::t_other);

    // https://hopstorawpointers.blogspot.com/2018/06/pybind11-and-python-sub-modules.html
    m.attr("__name__") = "regot._internal";
    m.doc() = "";
}

