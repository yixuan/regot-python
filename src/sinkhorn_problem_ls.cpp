#include "sinkhorn_problem.h"

namespace Sinkhorn {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RefVec = Eigen::Ref<Vector>;
using RefConstVec = Eigen::Ref<const Vector>;
using RefMat = Eigen::Ref<Matrix>;
using RefConstMat = Eigen::Ref<const Matrix>;

// Select a step size
double Problem::line_selection(
    const std::vector<double>& candid,
    const Vector& gamma, const Vector& direc, double curobj,
    Matrix& T, double& objval,
    bool verbose,
    std::ostream& cout
) const
{
    const int nc = static_cast<int>(candid.size());
    double best_step = 1.0;
    objval = std::numeric_limits<double>::infinity();
    Vector newgamma(gamma.size());
    for (int i = 0; i < nc; i++)
    {
        const double alpha = candid[i];
        newgamma.noalias() = gamma + alpha * direc;
        const double objfn = dual_obj(newgamma, T);
        if (objfn < objval)
        {
            best_step = alpha;
            objval = objfn;
        }
        if (objval < curobj)
        {
            return best_step;
        }
    }
    return best_step;
}

// Backtracking line search with Armijo conditions
double Problem::line_search_armijo(
    const Vector& gamma, const Vector& direc,
    double curobj, const Vector& curgrad,
    Matrix& T,
    double theta, double kappa,
    int max_iter, bool verbose,
    std::ostream& cout
) const
{
    // Initial step size
    double alpha = 1.0;
    // thresh < 0 if direc is a descent direction
    double thresh = theta * curgrad.dot(direc);
    Vector newgamma(gamma.size());
    for (int k = 0; k < max_iter; k++)
    {
        newgamma.noalias() = gamma + alpha * direc;
        const double newf = dual_obj(newgamma, T);
        if (newf <= curobj + alpha * thresh)
            break;
        alpha *= kappa;
    }
    return alpha;
}

// Backtracking line search with Wolfe conditions
double Problem::line_search_wolfe(
    const Vector& gamma, const Vector& direc, Matrix& T,
    double f, const Vector& g,
    double c1, double c2,
    int max_iter, bool verbose,
    std::ostream& cout
) const
{
    // Set up parameters for line search
    double alpha = 1.0;

    // Variables for line search
    double newf = std::numeric_limits<double>::infinity();
    Vector newgamma = gamma;
    Vector newg= g;

    // Backtracking line search
    int i;
    for (i = 0; i < max_iter; ++i)
    {
        newgamma.noalias() = gamma + alpha * direc;
        newf = dual_obj_grad(newgamma, newg, T, true);

        double dot_prod = g.dot(direc);
        if (newf > f + c1 * alpha * dot_prod)
        {
            // alpha too large, f value too high
            alpha *= 0.5;
        }
        else if (newg.dot(direc) < c2 * dot_prod)
        {
            // alpha too small, gradient too small (gradient is negative)
            alpha *= 2.1;
        }
        else
        {
            // condition satisfied
            break;
        }
    }
    return alpha;
}


}  // namespace Sinkhorn

