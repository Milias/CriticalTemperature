#pragma once

#include <iostream>
#include <chrono>
#include <utility>
#include <assert.h>
#include <thread>

#include <complex>

#include <cmath>

#include <omp.h>
#include <gmp.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_sf_zeta.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_dawson.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sum.h>

#include <mpfr.h>
#include <boost/numeric/odeint.hpp>

#include <arf.h>

#include <arb.h>
#include <arb_hypgeom.h>

#include <acb.h>

#include <optim/optim.hpp>

#include "utils.h"

/*** MPFR ***/

constexpr bool use_mpfr{false};
constexpr mp_prec_t prec{64};

/*** gsl constants ***/

constexpr double global_eps{1e-10};
constexpr int max_iter{64};

/*** Initialization ***/

void initializeMPFR_GSL();

/*** Utility functions ***/

double logExp(double x, double xmax = 1e3);
double logExp_mpfr(double x, double xmax = 1e3);

// real(Li_s(exp(z)))
double polylogExp(double s, double z);

// real(Li_s(-exp(z)))
double polylogExpM(double s, double z);

// find z such that PolyLog[s, Exp[z]] == a.
double invPolylogExp(double s, double a);

// find z such that -PolyLog[s, -Exp[z]] == a.
double invPolylogExpM(double s, double a);

// https://math.stackexchange.com/questions/712434/erfaib-error-function-separate-into-real-and-imaginary-part
double erf_fk(double x, double y, uint32_t k);
double erf_gk(double x, double y, uint32_t k);

double erf_sterm_r(double x, double y, double k);
double erf_sterm_i(double x, double y, double k);

// real(erf(x + i * y))
double erf_r(double x, double y, uint32_t n = 64, double eps = 1e-16);
// imag(erf(x + i * y))
double erf_i(double x, double y, uint32_t n = 64, double eps = 1e-16);

/*
 * Solve the equation
 *
 * y^2 / (1 - y) == v
 */
double y_eq_s(double v);

#ifndef SWIG

/*
 * Struct used when saving the full solution
 * for an ODE.
 */

template <typename state, typename time = double>
struct ode_observer {
  std::vector<state> & f_vec;
  std::vector<time> & t_vec;

  void operator()(const state & f, time t) {
    f_vec.push_back(f);
    t_vec.push_back(t);
  }
};

template <typename T> T constexpr sqrtNewtonRaphson(T x, T curr, T prev) {
  return curr == prev ? curr : sqrtNewtonRaphson<T>(x, 0.5 * (curr + x / curr), curr);
}

/*
* Constexpr version of the square root
* Return value:
*   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*
* https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
*/
template <typename T> T constexpr const_sqrt(T x) {
  return x >= 0 && x < std::numeric_limits<T>::infinity()
    ? sqrtNewtonRaphson<T>(x, x, 0)
    : std::numeric_limits<T>::quiet_NaN();
}

std::complex<double> erf_c(std::complex<double> & z);

template <typename T> T constexpr sign(T x) { return x == 0 ? 0 : (x > 0 ? 1 : -1); }
template <typename T> T constexpr kron(T a, T b) { return a == b ? 1 : 0; }

#endif

/*** Struve Function ***/

double struve(double v, double x);

