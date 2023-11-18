#include "qrot_problem.h"
#include "approx_proj.h"

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Compute the primal objective function
double Problem::primal_val(const Vector& gamma) const
{
    // Get transport plan
    Matrix plan = gamma.head(m_n).replicate(1, m_m) +
        gamma.tail(m_m).transpose().replicate(m_n, 1) -
        m_M;
    plan.noalias() = plan.cwiseMax(0.0) / m_reg;

    // Approximate projection
    Matrix plan_feas = approx_proj(plan, m_a, m_b);

    double prim_val = plan_feas.cwiseProduct(m_M).sum() +
            0.5 * m_reg * plan_feas.squaredNorm();
    return prim_val;
}



// Input gamma and M, compute statistics related to D:
//     ||D||_F^2, D row sum, D column sum
// D[i, j] = (alpha[i] + beta[j] - M[i, j])+, gamma = [alpha, beta]
//
// Define a function template to remove unnecessary computation
// at compile time
//
// Assume the memories of D_rowsum and D_colsum have been properly allocated
template <bool SquaredNorm = true, bool RowSum = true, bool ColSum = true>
double compute_D_stats(
    const Matrix& M, const double* alpha, const double* beta,
    double* D_rowsum, double* D_colsum
)
{
    // alpha = gamma[:n], beta = gamma[n:]
    const int n = static_cast<int>(M.rows());
    const int m = static_cast<int>(M.cols());
    const double* Mdata = M.data();

    // Initialization
    double D_sqnorm = 0.0;
    if (RowSum)
    {
        std::fill_n(D_rowsum, n, 0.0);
    }
    if (ColSum)
    {
        std::fill_n(D_colsum, m, 0.0);
    }

    for (int j = 0; j < m; j++)
    {
        const double betaj = beta[j];
        double colsumj = 0.0;
        for (int i = 0; i < n; i++, Mdata++)
        {
            const double Cij = alpha[i] + betaj - *Mdata;
            if (Cij > 0.0)
            {
                // C[i, j] > 0 => D[i, j] = C[i, j]
                //
                // 1. Add D[i, j]^2 to D_sqnorm
                // 2. Add D[i, j] to D_rowsum[i]
                // 3. Add D[i, j] to D_colsum[j]
                if (SquaredNorm)
                {
                    D_sqnorm += Cij * Cij;
                }
                if (RowSum)
                {
                    D_rowsum[i] += Cij;
                }
                if (ColSum)
                {
                    colsumj += Cij;
                }
            }
            // C[i, j] <= 0 => D[i, j] = 0
        }
        if (ColSum)
        {
            D_colsum[j] = colsumj;
        }
    }

    return D_sqnorm;
}

// Compute the objective function
// f(alpha, beta) = 0.5 * ||(alpha (+) beta - M)+||^2 - reg * (a' * alpha + b' * beta)
double Problem::dual_obj_vanilla(const Vector& gamma) const
{
    // Matrix C = gamma.head(m_n).replicate(1, m_m) +
    //     gamma.tail(m_m).transpose().replicate(m_n, 1) -
    //     m_M;
    // double obj = 0.5 * C.cwiseMax(0.0).squaredNorm() -
    //     m_reg * (gamma.head(m_n).dot(m_a) + gamma.tail(m_m).dot(m_b));

    const double D_sqnorm = compute_D_stats<true, false, false>(
        m_M, gamma.data(), gamma.data() + m_n, nullptr, nullptr);
    const double obj = 0.5 * D_sqnorm - m_reg * (
        gamma.head(m_n).dot(m_a) + gamma.tail(m_m).dot(m_b));
    return obj;
}
// SIMD version
double Problem::dual_obj(const Vector& gamma) const
{
    // Packet type
    using Eigen::Index;
    using Eigen::internal::ploadu;
    using Eigen::internal::pset1;
    using Eigen::internal::padd;
    using Eigen::internal::psub;
    using Eigen::internal::pmul;
    using Eigen::internal::pmax;
    using Eigen::internal::predux;
    using Packet = Eigen::internal::packet_traits<double>::type;
    constexpr unsigned char PacketSize = Eigen::internal::packet_traits<double>::size;
    constexpr unsigned char Peeling = 2;
    constexpr unsigned char Increment = Peeling * PacketSize;

    // Matrix C = gamma.head(m_n).replicate(1, m_m) +
    //     gamma.tail(m_m).transpose().replicate(m_n, 1) -
    //     m_M;
    // double obj = 0.5 * C.cwiseMax(0.0).squaredNorm() -
    //     m_reg * (gamma.head(m_n).dot(m_a) + gamma.tail(m_m).dot(m_b));

    // alpha = gamma[:n], beta = gamma[n:]
    const double* alpha = gamma.data();
    const double* beta = alpha + m_n;
    const double* M = m_M.data();

    // Decide the length of the inner loop
    // n % (2^k) == n & (2^k-1), see https://stackoverflow.com/q/3072665
    // const Index peeling_end = m_n - m_n % Increment;
    const Index aligned_end = m_n - (m_n & (PacketSize - 1));
    const Index peeling_end = m_n - (m_n & (Increment - 1));
    Packet zero = pset1<Packet>(0.0);

    double obj = 0.0;
    for (int j = 0; j < m_m; j++)
    {
        const double betaj = beta[j];
        Packet pbetaj = pset1<Packet>(betaj);
        alpha = gamma.data();
        for (Index i = 0; i < peeling_end; i += Increment)
        {
            Packet alphai0 = ploadu<Packet>(alpha);
            Packet alphai1 = ploadu<Packet>(alpha + PacketSize);
            Packet Mi0j = ploadu<Packet>(M);
            Packet Mi1j = ploadu<Packet>(M + PacketSize);
            Packet Ci0j = psub(padd(alphai0, pbetaj), Mi0j);
            Packet Ci1j = psub(padd(alphai1, pbetaj), Mi1j);
            Packet Di0j = pmax(Ci0j, zero);
            Packet Di1j = pmax(Ci1j, zero);
            obj += predux(pmul(Di0j, Di0j)) + predux(pmul(Di1j, Di1j));

            alpha += Increment;
            M += Increment;
        }
        if (aligned_end != peeling_end)
        {
            Packet alphai = ploadu<Packet>(alpha);
            Packet Mij = ploadu<Packet>(M);
            Packet Cij = psub(padd(alphai, pbetaj), Mij);
            Packet Dij = pmax(Cij, zero);
            obj += predux(pmul(Dij, Dij));

            alpha += PacketSize;
            M += PacketSize;
        }
        // Remaining elements
        for (Index i = aligned_end; i < m_n; i++, alpha++, M++)
        {
            const double Cij = *alpha + betaj - *M;
            obj += (Cij > 0.0) ? (Cij * Cij) : (0.0);
        }
    }
    obj = 0.5 * obj - m_reg * (gamma.head(m_n).dot(m_a) + gamma.tail(m_m).dot(m_b));
    return obj;
}

// Compute the gradient
void Problem::dual_grad(const Vector& gamma, Vector& grad) const
{
    // g(alpha) = D * 1m - reg * a
    // g(beta) = D' * 1n - reg * b
    grad.resize(m_n + m_m);

    // First compute D * 1m and D' * 1n
    double* D_rowsum = grad.data();
    double* D_colsum = D_rowsum + m_n;
    compute_D_stats<false, true, true>(
        m_M, gamma.data(), gamma.data() + m_n, D_rowsum, D_colsum);

    // grad now contains [D * 1m, D' * 1n]
    grad.head(m_n).noalias() -= m_reg * m_a;
    grad.tail(m_m).noalias() -= m_reg * m_b;
}

// Compute the objective function and gradient
double Problem::dual_obj_grad(const Vector& gamma, Vector& grad) const
{
    // g(alpha) = D * 1m - reg * a
    // g(beta) = D' * 1n - reg * b
    grad.resize(m_n + m_m);

    // First compute D * 1m and D' * 1n
    double* D_rowsum = grad.data();
    double* D_colsum = D_rowsum + m_n;
    const double D_sqnorm = compute_D_stats<true, true, true>(
        m_M, gamma.data(), gamma.data() + m_n, D_rowsum, D_colsum);

    // grad now contains [D * 1m, D' * 1n]
    grad.head(m_n).noalias() -= m_reg * m_a;
    grad.tail(m_m).noalias() -= m_reg * m_b;

    const double obj = 0.5 * D_sqnorm - m_reg * (
        gamma.head(m_n).dot(m_a) + gamma.tail(m_m).dot(m_b));
    return obj;
}

// Compute the objective function, gradient, and generalized Hessian
// C = alpha (+) beta - M, D = (C)+, sigma = ifelse(C >= 0, 1, 0)
// f(alpha, beta) = 0.5 * ||D||^2 - reg * (a' * alpha + b' * beta)
// g(alpha) = D * 1m - reg * a
// g(beta) = D' * 1n - reg * b
// H = [diag(sigma * 1m)              sigma]
//     [          sigma'  diag(sigma' * 1n)]
void Problem::dual_obj_grad_hess(
    const Vector& gamma, double& obj, Vector& grad, Hessian& hess
) const
{
    double D_sqnorm = 0.0;
    grad.resize(m_n + m_m);
    double* D_rowsum = grad.data();
    double* D_colsum = D_rowsum + m_n;

    // Sparse Hessian
    hess.compute_D_hess(gamma, m_M, D_sqnorm, D_rowsum, D_colsum);

    // Objective function
    obj = 0.5 * D_sqnorm - m_reg * (
        gamma.head(m_n).dot(m_a) + gamma.tail(m_m).dot(m_b));

    // Gradient
    // After calling hess.compute_D_hess(), grad now contains [D * 1m, D' * 1n]
    grad.head(m_n).noalias() -= m_reg * m_a;
    grad.tail(m_m).noalias() -= m_reg * m_b;
}

// Select a step size
double Problem::line_selection(
    const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
    double& objval, bool verbose, std::ostream& cout
) const
{
    // std::vector<double> objfns;
    // objfns.reserve(candid.size() + 1);
    // Vector newgamma(gamma.size());
    // for(auto alpha: candid)
    // {
    //     newgamma.noalias() = gamma + alpha * direc;
    //     double objfn = dual_obj(newgamma);
    //     objfns.push_back(objfn);
    // }

    // Parallel search
    const int nc = static_cast<int>(candid.size());
    std::vector<double> objfns(nc);
    #pragma omp parallel for shared(gamma) num_threads(nc) schedule(static)
    for (int i = 0; i < nc; i++)
    {
        Vector newgamma = gamma + candid[i] * direc;
        double objfn = dual_obj(newgamma);
        objfns[i] = objfn;
    }

    // https://stackoverflow.com/a/65163939
    std::vector<double>::iterator min_loc = std::min_element(objfns.begin(), objfns.end());
    std::size_t ind = std::distance(objfns.begin(), min_loc);
    double alpha = candid[ind];
    objval = objfns[ind];

    if (verbose)
    {
        cout << "alpha = " << alpha << ", line search: [";
        for(auto objfn: objfns)
            cout << objfn << ", ";
        cout << "]" << std::endl << std::endl;
    }

    return alpha;
}

// Select a step size
double Problem::line_selection2(
    const std::vector<double>& candid, const Vector& gamma, const Vector& direc,
    double curobj, double& objval, bool verbose, std::ostream& cout
) const
{
    const int nc = static_cast<int>(candid.size());
    double best_step = 1.0;
    objval = std::numeric_limits<double>::infinity();
    for (int i = 0; i < nc; i++)
    {
        const double alpha = candid[i];
        Vector newgamma = gamma + alpha * direc;
        const double objfn = dual_obj(newgamma);
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



// Piecewise linear equation
//
// f(x) = sum_j (x - y[j])+ = b, b > 0
inline double piecewise_linear_equation(const Vector& y, double b)
{
    const int n = y.size();

    // Sort y
    Vector ysort = y;
    double* ys = ysort.data();
    std::sort(ys, ys + n);

    double ys_cumsum = 0.0, f_ys = 0.0;
    for (int i = 0; i < n - 1; i++)
    {
        ys_cumsum += ys[i];
        const double f_ys_next = (i + 1) * ys[i + 1] - ys_cumsum;
        if (b >= f_ys && b < f_ys_next)
        {
            const double x = (b + ys_cumsum) / (i + 1);
            return x;
        }
        f_ys = f_ys_next;
    }

    // b >= f(ys[n - 1])
    ys_cumsum += ys[n - 1];
    const double x = (b + ys_cumsum) / n;
    return x;
}

// Optimal beta given alpha
void Problem::optimal_beta(const RefConstVec& alpha, RefVec beta) const
{
    // beta.resize(m_m);
    Vector y(m_n);
    for (int j = 0; j < m_m; j++)
    {
        y.noalias() = m_M.col(j) - alpha;
        beta[j] = piecewise_linear_equation(y, m_reg * m_b[j]);
    }
}

// Optimal alpha given beta
void Problem::optimal_alpha(const RefConstVec& beta, RefVec alpha) const
{
    // alpha.resize(m_n);
    Vector y(m_m);
    for (int i = 0; i < m_n; i++)
    {
        y.noalias() = m_M.row(i).transpose() - beta;
        alpha[i] = piecewise_linear_equation(y, m_reg * m_a[i]);
    }
}

// Compute the objective function and gradient of semi-dual
double Problem::semi_dual_obj_grad(const Vector& alpha, Vector& grad) const
{
    grad.resize(m_n);
    Vector beta(m_m);
    optimal_beta(alpha, beta);

    // First compute D * 1m
    double* D_rowsum = grad.data();
    const double D_sqnorm = compute_D_stats<true, true, false>(
        m_M, alpha.data(), beta.data(), D_rowsum, nullptr);

    // Objective function
    const double obj = 0.5 * D_sqnorm - m_reg * (
        alpha.dot(m_a) + beta.dot(m_b));

    // Gradient
    // grad now contains D * 1m
    grad.noalias() -= m_reg * m_a;

    return obj;
}
