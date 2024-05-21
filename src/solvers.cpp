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

QROTResult qrot_apdagd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_apdagd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_assn(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_assn_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_bcd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_bcd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_gd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_gd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_grssn(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, double shift = 0.001,
    bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    solver_opts.shift = shift;
    QROT::qrot_grssn_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_lbfgs_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_lbfgs_dual_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_lbfgs_semi_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_lbfgs_semi_dual_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_pdaam(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_pdaam_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_s5n(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    QROTSolverOpts solver_opts;
    QROT::qrot_s5n_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}



SinkhornResult sinkhorn_apdagd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    Sinkhorn::sinkhorn_apdagd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_bcd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    Sinkhorn::sinkhorn_bcd_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_lbfgs_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    Sinkhorn::sinkhorn_lbfgs_dual_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_newton(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    Sinkhorn::sinkhorn_newton_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}

SinkhornResult sinkhorn_ssns(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    SinkhornResult result;
    SinkhornSolverOpts solver_opts;
    solver_opts.method = 1;
    Sinkhorn::sinkhorn_ssns_internal(
        result, M, a, b, reg, solver_opts, tol, max_iter,
        verbose, std::cout);

    return result;
}



PYBIND11_MODULE(_internal, m) {
    // Solvers
    m.def("qrot_apdagd", &qrot_apdagd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_assn", &qrot_assn,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_bcd", &qrot_bcd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_gd", &qrot_gd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_grssn", &qrot_grssn,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "shift"_a = 0.001,
        "verbose"_a = false);
    m.def("qrot_lbfgs_dual", &qrot_lbfgs_dual,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_lbfgs_semi_dual", &qrot_lbfgs_semi_dual,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_pdaam", &qrot_pdaam,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("qrot_s5n", &qrot_s5n,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    
    m.def("sinkhorn_apdagd", &sinkhorn_apdagd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("sinkhorn_bcd", &sinkhorn_bcd,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("sinkhorn_lbfgs_dual", &sinkhorn_lbfgs_dual,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("sinkhorn_newton", &sinkhorn_newton,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);
    m.def("sinkhorn_ssns", &sinkhorn_ssns,
        "M"_a, "a"_a, "b"_a, "reg"_a,
        "tol"_a = 1e-6, "max_iter"_a = 1000, "verbose"_a = false);

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
        .def_readwrite("prim_vals", &SinkhornResult::prim_vals)
        .def_readwrite("mar_errs", &SinkhornResult::mar_errs)
        .def_readwrite("run_times", &SinkhornResult::run_times)
        .def_readwrite("density", &SinkhornResult::density);

    // https://hopstorawpointers.blogspot.com/2018/06/pybind11-and-python-sub-modules.html
    m.attr("__name__") = "regot._internal";
    m.doc() = "";
}
