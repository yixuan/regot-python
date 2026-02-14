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
    constexpr double deltal = 1.1, deltau = 0.66;
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
            (std::min)(at + deltau * (au - at), res) :
            (std::max)(at + deltau * (au - at), res);
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
        (std::min)(at + deltau * (au - at), ae) :
        (std::max)(at + deltau * (au - at), ae);
}

// More-Thuente line search with Wolfe conditions
double Problem::line_search_wolfe(
    double init_step,
    const Vector& gamma, const Vector& direc,
    double curobj, const Vector& curgrad,
    Matrix& T, bool& recompute_T,
    double c1, double c2,
    int max_iter, bool verbose,
    std::ostream& cout
) const
{
    using std::abs;

    // Typically the T matrix have been computed on the new point
    // when line search exits, but there are cases that a
    // different step is returned
    // In such cases, we flag recompute_T = true
    recompute_T = false;

    // Initial step size
    constexpr double step_min = 1e-10, step_max = 2.0;
    init_step = (std::max)(init_step, step_min);
    init_step = (std::min)(init_step, step_max);
    double step = init_step;

    // Make copies
    Vector x = gamma, grad = curgrad;
    double fx = curobj, dg = curgrad.dot(direc);

    // Save the function value at the current x
    const double fx_init = curobj;
    // Projection of gradient on the search direction
    const double dg_init = dg;
    // Make sure d points to a descent direction
    if (dg_init > 0.0)
    {
        recompute_T = true;
        return init_step;
    }

    // Tolerance for convergence test
    // Sufficient decrease
    const double test_decr = c1 * dg_init;
    // Curvature
    const double test_curv = -c2 * dg_init;

    // The bracketing interval
    constexpr double Inf = std::numeric_limits<double>::infinity();
    double I_lo = 0.0, I_hi = Inf;
    double fI_lo = 0.0, fI_hi = Inf;
    double gI_lo = (1.0 - c1) * dg_init, gI_hi = Inf;
    double psiI_lo = fI_lo;
    double fx_lo = fx_init, dg_lo = dg_init;

    // Status variables
    bool bracketed = false;
    bool f_is_psi = true;
    bool use_step_min_safeguard = (step_min > 0.0);
    double I_width = Inf;
    double I_width_prev = Inf;
    int I_shrink_fail_count = 0;

    // Constants
    constexpr double delta_max = 1.1;
    constexpr double delta_min = 7.0 / 12;
    constexpr double shrink = 0.66;
    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // Function value and gradient at the current step size
        x.noalias() = gamma + step * direc;
        fx = dual_obj_grad(x, grad, T, true);
        dg = grad.dot(direc);

        // phi(step) = f(xp + step * drt) = fx
        // phi'(step) = g(xp + step * drt)^T d = dg
        // psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
        // psi'(step) = dg - test_decr
        const double psit = fx - fx_init - step * test_decr;
        const double dpsit = dg - test_decr;

        // Convergence test
        if (psit <= 0.0 && abs(dg) <= test_curv)
        {
            return step;
        }

        // Test whether step hits the boundaries and satisfies the exit conditions
        if (step <= step_min && (psit > 0.0 || dpsit >= 0.0))
        {
            return step;
        }
        if (step >= step_max && (psit <= 0.0 && dpsit < 0.0))
        {
            return step;
        }

        // Check and update the status of f_is_psi
        // f is initially set to psi, and is changed to phi in
        // subsequent iterations if psi(step) <= 0 and phi'(step) >= 0
        //
        // NOTE: empirically we find that using psi is usually better,
        //       so for now we do not follow the implementation of [1]
        /*
        if (f_is_psi && (psit <= 0.0 && dg >= 0.0))
        {
            f_is_psi = false;
        }
        */
        const double ft = f_is_psi ? psit : fx;
        const double gt = f_is_psi ? dpsit : dg;

        // Check and update the status of use_step_min_safeguard
        // We impose a safeguarding rule that guarantees testing
        // step_min if psi(alpha_k) > 0 or psi'(alpha_k) >= 0
        // holds from the beginning
        if (use_step_min_safeguard && (psit <= 0.0 && dpsit < 0.0))
        {
            use_step_min_safeguard = false;
        }

        // Update new step
        double new_step;
        const bool in_case_2 = (psit <= psiI_lo) && (dpsit * (I_lo - step) > 0.0);
        if (in_case_2)
        {
            // For Case 2, we apply the safeguarding rule
            // newat = min(at + delta * (at - al), amax), delta in [1.1, 4]
            new_step = (std::min)(step_max, step + delta_max * (step - I_lo));
        }
        else
        {
            // For Case 1 and Case 3, use information of f and g to select new step
            new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
            // Force new step in [step_min, step_max]
            new_step = (std::max)(new_step, step_min);
            new_step = (std::min)(new_step, step_max);

            // Apply safeguarding rule related to step_min when necessary:
            //     step+ in [alpha_min, max{delta_min * step, alpha_min}]
            //
            // If use_step_min_safeguard = true, then new_step cannot be obtained
            // from Case 2, since in Case 2 we have
            //     psi(alpha_k) <= 0 and psi'(alpha_k) < 0
            if (use_step_min_safeguard)
            {
                const double lower = step_min;
                const double upper = (std::max)(step_min, delta_min * step);
                new_step = (std::max)(new_step, lower);
                new_step = (std::min)(new_step, upper);
            }
        }

        // Update bracketing interval
        if (psit > psiI_lo)
        {
            // Case 1: psi(step) > psi(I_lo)
            I_hi = step;
            fI_hi = ft;
            gI_hi = gt;
        }
        else if (in_case_2)
        {
            // Case 2: psi(step) <= psi(I_lo), psi'(step)(I_lo - step) > 0
            I_lo = step;
            fI_lo = ft;
            gI_lo = gt;
            psiI_lo = psit;
            fx_lo = fx;
            dg_lo = dg;
        }
        else
        {
            // Case 3: psi(step) <= psi(I_lo), psi'(step)(I_lo - step) <= 0
            I_hi = I_lo;
            fI_hi = fI_lo;
            gI_hi = gI_lo;

            I_lo = step;
            fI_lo = ft;
            gI_lo = gt;
            psiI_lo = psit;
            fx_lo = fx;
            dg_lo = dg;
        }

        // Check and update the status of bracketed
        // bracketed is true if we have entered Case 1 or Case 3,
        // and I is contained in [step_min, step_max]
        if ((!bracketed) && (!in_case_2))
        {
            const double I_left = (std::min)(I_lo, I_hi);
            const double I_right = (std::max)(I_lo, I_hi);
            bracketed = (I_left >= step_min && I_right <= step_max);
        }

        // If bracketed, enforce sufficient interval shrink; if not shrinking enough, use bisection
        if (bracketed)
        {
            I_width_prev = I_width;
            I_width = abs(I_hi - I_lo);
            // Test interval shrinkage
            if (I_width_prev < Inf && I_width > shrink * I_width_prev)
            {
                I_shrink_fail_count += 1;
            }
            else
            {
                I_shrink_fail_count = 0;
            }
            // If interval fails to shrink enough twice, select new_step using bisection
            if (I_shrink_fail_count >= 2)
            {
                new_step = (I_lo + I_hi) / 2.0;
                I_shrink_fail_count = 0;
            }
        }

        // Set the new_step
        step = new_step;
    }

    // If we have used up all line search iterations, then the strong Wolfe condition
    // is not met. We choose not to raise an exception, but to return the best
    // step size so far

    // First test whether the last step is better than I_lo
    // If yes, return the last step
    const double psit = fx - fx_init - step * test_decr;
    if (psit <= psiI_lo)
        return step;

    // If not, then the best step size so far is I_lo, but it needs to be positive
    if (I_lo <= 0.0)
    {
        recompute_T = true;
        return init_step;
    }

    // Return everything with _lo
    step = I_lo;
    fx = fx_lo;
    dg = dg_lo;
    recompute_T = true;
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
