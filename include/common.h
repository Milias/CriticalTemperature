#pragma once

#include <iostream>
#include <chrono>
#include <assert.h>

#include <complex>

#include <cmath>
#include <gmp.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_sf_zeta.h>
#include <gsl/gsl_deriv.h>
#include <mpfr.h>
#include <boost/numeric/odeint.hpp>

#include <arf.h>
#include <arb.h>
#include <acb.h>

/*** MPFR ***/

constexpr bool use_mpfr = false;
constexpr mp_prec_t prec = 64;

/*** gsl_integration workspace size ***/

constexpr size_t w_size = 1<<3;

extern "C" {
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
}

std::complex<double> erf_c(std::complex<double> & z);

double invPolylogExp_f(double z, void * params);
double invPolylogExp_df(double z, void * params);
double invPolylogExpM_f(double z, void * params);
double invPolylogExpM_df(double z, void * params);

int rosenbrock_f(const gsl_vector * x, void * params, gsl_vector * f);
double * solve_rosenbrock(double a, double b);

