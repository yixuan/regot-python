#ifndef REGOT_PDIP_DENSE_CHOL_H
#define REGOT_PDIP_DENSE_CHOL_H

#include <vector>

bool pdip_cholesky_lower(int n, std::vector<double>& A);
void pdip_cholesky_solve(int n, const std::vector<double>& L, const double* b, double* x);
void pdip_cholesky_solve(int n, const std::vector<double>& L, const double* b, double* x, double* work);

#endif
