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

// Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
// that interpolates fa, ga, and fb, assuming the minimizer exists
// For case I: fb >= fa and ga * (b - a) < 0
inline double quadratic_minimizer(double a, double b, double fa, double ga, double fb)
{
    const double ba = b - a;
    const double w = 0.5 * ba * ga / (fa - fb + ba * ga);
    return a + w * ba;
}

// Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
// that interpolates fa, ga and gb, assuming the minimizer exists
// The result actually does not depend on fa
// For case II: ga * (b - a) < 0, ga * gb < 0
// For case III: ga * (b - a) < 0, ga * ga >= 0, |gb| <= |ga|
inline double quadratic_minimizer(double a, double b, double ga, double gb)
{
    const double w = ga / (ga - gb);
    return a + w * (b - a);
}

// Local minimizer of a cubic function q(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
// that interpolates fa, ga, fb and gb, assuming a != b
// Also sets a flag indicating whether the minimizer exists
inline double cubic_minimizer(double a, double b, double fa, double fb,
                              double ga, double gb, bool& exists)
{
    using std::abs;
    using std::sqrt;

    const double apb = a + b;
    const double ba = b - a;
    const double ba2 = ba * ba;
    const double fba = fb - fa;
    const double gba = gb - ga;
    // z3 = c3 * (b-a)^3, z2 = c2 * (b-a)^3, z1 = c1 * (b-a)^3
    const double z3 = (ga + gb) * ba - 2.0 * fba;
    const double z2 = 0.5 * (gba * ba2 - 3.0 * apb * z3);
    const double z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;

    // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
    constexpr double eps = std::numeric_limits<double>::epsilon();
    if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1))
    {
        // Minimizer exists if c2 > 0
        exists = (z2 * ba > 0.0);
        // Return the end point if the minimizer does not exist
        return exists ? (-0.5 * z1 / z2) : b;
    }

    // Now we can assume z3 > 0
    // The minimizer is a solution to the equation c1 + 2*c2 * x + 3*c3 * x^2 = 0
    // roots = -(z2/z3) / 3 (+-) sqrt((z2/z3)^2 - 3 * (z1/z3)) / 3
    //
    // Let u = z2/(3z3) and v = z1/z2
    // The minimizer exists if v/u <= 1
    const double u = z2 / (3.0 * z3), v = z1 / z2;
    const double vu = v / u;
    exists = (vu <= 1.0);
    if (!exists)
        return b;

    // We need to find a numerically stable way to compute the roots, as z3 may still be small
    //
    // If |u| >= |v|, let w = 1 + sqrt(1-v/u), and then
    // r1 = -u * w, r2 = -v / w, r1 does not need to be the smaller one
    //
    // If |u| < |v|, we must have uv <= 0, and then
    // r = -u (+-) sqrt(delta), where
    // sqrt(delta) = sqrt(|u|) * sqrt(|v|) * sqrt(1-u/v)
    double r1 = 0.0, r2 = 0.0;
    if (abs(u) >= abs(v))
    {
        const double w = 1.0 + sqrt(1.0 - vu);
        r1 = -u * w;
        r2 = -v / w;
    }
    else
    {
        const double sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1 - u / v);
        r1 = -u - sqrtd;
        r2 = -u + sqrtd;
    }
    return (z3 * ba > 0.0) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
}

// Select the next step size according to the current step sizes,
// function values, and derivatives
inline double step_selection(
    double al, double au, double at,
    double fl, double fu, double ft,
    double gl, double gu, double gt)
{
    using std::abs;

    if (al == au)
        return al;

    // If ft = Inf or gt = Inf, we return the middle point of al and at
    if (!std::isfinite(ft) || !std::isfinite(gt))
        return (al + at) / 2.0;

    // ac: cubic interpolation of fl, ft, gl, gt
    // aq: quadratic interpolation of fl, gl, ft
    bool ac_exists;
    const double ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
    const double aq = quadratic_minimizer(al, at, fl, gl, ft);
    // Case 1: ft > fl
    if (ft > fl)
    {
        // This should not happen if ft > fl, but just to be safe
        if (!ac_exists)
            return aq;
        // Then use the scheme described in the paper
        return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / 2.0);
    }

    // as: quadratic interpolation of gl and gt
    const double as = quadratic_minimizer(al, at, gl, gt);
    // Case 2: ft <= fl, gt * gl < 0
    if (gt * gl < 0.0)
        return (abs(ac - at) >= abs(as - at)) ? ac : as;

    // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
    const double deltal = 1.1, deltau = 0.66;
    if (abs(gt) < abs(gl))
    {
        // We choose either ac or as
        // The case for ac: 1. It exists, and
        //                  2. ac is farther than at from al, and
        //                  3. ac is closer to at than as
        // Cases for as: otherwise
        const bool choose_ac = ac_exists &&
            ((ac - at) * (at - al) > 0.0) &&
            (abs(ac - at) < abs(as - at));
        const double res = choose_ac ? ac : as;
        // Postprocessing the chosen step
        return (at > al) ?
            std::min(at + deltau * (au - at), res) :
            std::max(at + deltau * (au - at), res);
    }

    // Simple extrapolation if au, fu, or gu is infinity
    if ((!std::isfinite(au)) || (!std::isfinite(fu)) || (!std::isfinite(gu)))
        return at + deltal * (at - al);

    // ae: cubic interpolation of ft, fu, gt, gu
    bool ae_exists;
    const double ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
    // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
    // The following is not used in the paper, but it seems to be a reasonable safeguard
    return (at > al) ?
        std::min(at + deltau * (au - at), ae) :
        std::max(at + deltau * (au - at), ae);
}

double Problem::line_search_wolfe(
    double init_step,
    const Vector& gamma, const Vector& direc,
    double curobj, const Vector& curgrad,
    Matrix& T,
    double c1, double c2,
    int max_iter, bool verbose,
    std::ostream& cout
) const
{
    // Initial step size
    double step = init_step, step_max = 2.0;

    Vector x = gamma, grad = curgrad;
    double fx = curobj, dg = curgrad.dot(direc);

    // Save the function value at the current x
    const double fx_init = curobj;
    // Projection of gradient on the search direction
    const double dg_init = dg;
    // Make sure d points to a descent direction
    if (dg_init > 0.0)
        return step;

    // Tolerance for convergence test
    // Sufficient decrease
    const double test_decr = c1 * dg_init;
    // Curvature
    const double test_curv = -c2 * dg_init;

    // The bracketing interval
    double I_lo = 0.0, I_hi = std::numeric_limits<double>::infinity();
    double fI_lo = 0.0, fI_hi = std::numeric_limits<double>::infinity();
    double gI_lo = (1.0 - c1) * dg_init, gI_hi = std::numeric_limits<double>::infinity();
    double fx_lo = fx_init, dg_lo = dg_init;


    // Evaluate the current step size
    x.noalias() = gamma + step * direc;
    fx = dual_obj_grad(x, grad, T, true);
    dg = grad.dot(direc);

    // Convergence test
    if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
    {
        return step;
    }

    // Extrapolation factor
    constexpr double delta = 1.1;
    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // ft = psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
        // gt = psi'(step) = dg - mu * dg_init
        // mu = c1
        const double ft = fx - fx_init - step * test_decr;
        const double gt = dg - c1 * dg_init;

        // Update step size and bracketing interval
        double new_step;
        if (ft > fI_lo)
        {
            // Case 1: ft > fl
            new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
            // Sanity check: if the computed new_step is too small, typically due to
            // extremely large value of ft, switch to the middle point
            if (new_step <= 1e-12)
                new_step = (I_lo + step) / 2.0;

            I_hi = step;
            fI_hi = ft;
            gI_hi = gt;
        }
        else if (gt * (I_lo - step) > 0.0)
        {
            // Case 2: ft <= fl, gt * (al - at) > 0
            //
            // Page 291 of Moré and Thuente (1994) suggests that
            // newat = min(at + delta * (at - al), amax), delta in [1.1, 4]
            new_step = std::min(step_max, step + delta * (step - I_lo));

            I_lo = step;
            fI_lo = ft;
            gI_lo = gt;
            fx_lo = fx;
            dg_lo = dg;
        }
        else
        {
            // Case 3: ft <= fl, gt * (al - at) <= 0
            new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);

            I_hi = I_lo;
            fI_hi = fI_lo;
            gI_hi = gI_lo;

            I_lo = step;
            fI_lo = ft;
            gI_lo = gt;
            fx_lo = fx;
            dg_lo = dg;
        }

        // Case 1 and 3 are interpolations, whereas Case 2 is extrapolation
        // This means that Case 2 may return new_step = step_max,
        // and we need to decide whether to accept this value
        // 1. If both step and new_step equal to step_max, it means
        //    step will have no further change, so we accept it
        // 2. Otherwise, we need to test the function value and gradient
        //    on step_max, and decide later

        // In case step, new_step, and step_max are equal, directly return the computed x and fx
        if (step == step_max && new_step >= step_max)
        {
            return step;
        }
        // Otherwise, recompute x and fx based on new_step
        step = new_step;

        if (step < 1e-12 || step > 1e12)
            return init_step;

        // Update parameter, function value, and gradient
        x.noalias() = gamma + step * direc;
        fx = dual_obj_grad(x, grad, T, true);
        dg = grad.dot(direc);

        // Convergence test
        if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
        {
            return step;
        }

        // Now assume step = step_max, and we need to decide whether to
        // exit the line search (see the comments above regarding step_max)
        // If we reach here, it means this step size does not pass the convergence
        // test, so either the sufficient decrease condition or the curvature
        // condition is not met yet
        //
        // Typically the curvature condition is harder to meet, and it is
        // possible that no step size in [0, step_max] satisfies the condition
        //
        // But we need to make sure that its psi function value is smaller than
        // the best one so far. If not, go to the next iteration and find a better one
        if (step >= step_max)
        {
            const double ft_bound = fx - fx_init - step * test_decr;
            if (ft_bound <= fI_lo)
            {
                return step;
            }
        }

        // If we have used up all line search iterations, then the strong Wolfe condition
        // is not met. We choose not to raise an exception (unless no step satisfying
        // sufficient decrease is found), but to return the best step size so far
        if (iter >= max_iter)
        {
            // First test whether the last step is better than I_lo
            // If yes, return the last step
            const double ft = fx - fx_init - step * test_decr;
            if (ft <= fI_lo)
                return step;

            // If not, then the best step size so far is I_lo, but it needs to be positive
            if (I_lo <= 0.0)
                return init_step;

            // Return everything with _lo
            step = I_lo;
            fx = fx_lo;
            dg = dg_lo;
            return step;
        }
    }

    return step;
}



// Old implementation: backtracking line search with Wolfe conditions
/*
double Problem::line_search_wolfe(
    const Vector& gamma, const Vector& direc,
    double curobj, const Vector& curgrad,
    Matrix& T,
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
    Vector newg = curgrad;
    // dot_prod < 0 if direc is a descent direction
    double dot_prod = curgrad.dot(direc);

    // Backtracking line search
    int i;
    for (i = 0; i < max_iter; ++i)
    {
        newgamma.noalias() = gamma + alpha * direc;
        newf = dual_obj_grad(newgamma, newg, T, true);

        if (newf > curobj + c1 * alpha * dot_prod)
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
*/


}  // namespace Sinkhorn
