#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "qrot_solvers.h"
#include "sinkhorn_solvers.h"

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
    // Used in sparse Newton
    if (kwargs.contains("density"))
    {
        solver_opts.density = py::float_(kwargs["density"]);
    }
    if (kwargs.contains("shift"))
    {
        solver_opts.shift = py::float_(kwargs["shift"]);
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
        .def_readwrite("density", &SinkhornResult::density);

    // https://hopstorawpointers.blogspot.com/2018/06/pybind11-and-python-sub-modules.html
    m.attr("__name__") = "regot._internal";
    m.doc() = "";
}
