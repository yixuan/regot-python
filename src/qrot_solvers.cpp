#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "qrot_solvers.h"

namespace py = pybind11;
using namespace pybind11::literals;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefConstVec = Eigen::Ref<const Vector>;
using RefConstMat = Eigen::Ref<const Matrix>;

QROTResult qrot_apdagd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_apdagd_internal(
        result, M, a, b, reg, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_assn(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_assn_internal(
        result, M, a, b, reg, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_bcd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_bcd_internal(
        result, M, a, b, reg, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_gd(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_gd_internal(
        result, M, a, b, reg, tol, max_iter,
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
    qrot_grssn_internal(
        result, M, a, b, reg, tol, max_iter, shift,
        verbose, std::cout);

    return result;
}

QROTResult qrot_lbfgs_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_lbfgs_dual_internal(
        result, M, a, b, reg, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_lbfgs_semi_dual(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_lbfgs_semi_dual_internal(
        result, M, a, b, reg, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_pdaam(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_pdaam_internal(
        result, M, a, b, reg, tol, max_iter,
        verbose, std::cout);

    return result;
}

QROTResult qrot_s5n(
    RefConstMat M, RefConstVec a, RefConstVec b, double reg,
    double tol = 1e-6, int max_iter = 1000, bool verbose = false
)
{
    QROTResult result;
    qrot_s5n_internal(
        result, M, a, b, reg, tol, max_iter,
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

    // Returned object
    py::class_<QROTResult>(m, "qrot_result")
        .def(py::init<>())
        .def_readwrite("niter", &QROTResult::niter)
        .def_readwrite("plan", &QROTResult::plan)
        .def_readwrite("obj_vals", &QROTResult::obj_vals)
        .def_readwrite("mar_errs", &QROTResult::mar_errs)
        .def_readwrite("run_times", &QROTResult::run_times);

    // https://hopstorawpointers.blogspot.com/2018/06/pybind11-and-python-sub-modules.html
    m.attr("__name__") = "regot._internal";
    m.doc() = "";
}
