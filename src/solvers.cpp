// Role of this file:
// 1) Bind C++ solvers to Python;
// 2) Parse kwargs into per-algorithm options in one place;
// 3) Keep regot return-object fields consistent across the public API.
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
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
    // cg_max_iter: max inner CG iterations
    if (kwargs.contains("cg_max_iter"))
    {
        solver_opts.cg_max_iter = py::int_(kwargs["cg_max_iter"]);
    }
    // fixed_threshold: FP sparsity threshold; gates whether B2 / fixed-point path is used
    if (kwargs.contains("fixed_threshold"))
    {
        solver_opts.fixed_threshold = py::float_(kwargs["fixed_threshold"]);
    }
    // fp_max_iter: max inner fixed-point iterations for FP
    if (kwargs.contains("fp_max_iter"))
    {
        solver_opts.fp_max_iter = py::int_(kwargs["fp_max_iter"]);
    }
    // fp_exit_scale: inner FP stopping threshold scale
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

// Unified PDIP entry: naming aligned with QROT; inner_solver selects CG or FP (sparse Cholesky inner path for FP).
static PDIPResult run_pdip(
    bool use_fp_inner,
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    PDIPResult result;
    PDIPSolverOpts solver_opts;
    parse_pdip_opts(solver_opts, kwargs);
    Matrix M_T = M.transpose();
    if (use_fp_inner) {
        PDIP::pdip_fp_internal(
            result, M_T, a, b, reg, solver_opts, tol, max_iter,
            verbose, std::cout);
    } else {
        PDIP::pdip_cg_internal(
            result, M_T, a, b, reg, solver_opts, tol, max_iter,
            verbose, std::cout);
    }
    return result;
}

PDIPResult qrot_pdip(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    std::string mode = "cg";
    if (kwargs.contains("inner_solver"))
    {
        py::object v = kwargs["inner_solver"];
        if (!py::isinstance<py::str>(v))
        {
            throw std::runtime_error("qrot_pdip: inner_solver must be a string, e.g. \"cg\" or \"fp\"");
        }
        mode = py::str(v).cast<std::string>();
    }
    for (char &c : mode)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (mode != "cg" && mode != "fp")
    {
        throw std::runtime_error("qrot_pdip: inner_solver must be \"cg\" or \"fp\"");
    }
    return run_pdip(mode == "fp", M, a, b, reg, tol, max_iter, verbose, kwargs);
}

PDIPResult pdip_cg(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    return run_pdip(false, M, a, b, reg, tol, max_iter, verbose, kwargs);
}

PDIPResult pdip_fp(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol, int max_iter, int verbose, const py::kwargs &kwargs
)
{
    return run_pdip(true, M, a, b, reg, tol, max_iter, verbose, kwargs);
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
    // PDIP: unified entry qrot_pdip(inner_solver="cg"|"fp"); pdip_cg / pdip_fp are compatibility aliases.
    m.def("qrot_pdip", &qrot_pdip,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-8, "max_iter"_a = 1000, "verbose"_a = 0);
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

