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
#include <mpfr.h>

/*** MPFR ***/

constexpr bool use_mpfr = false;
constexpr mp_prec_t prec = 256;

/*** gsl_integration workspace size ***/

constexpr size_t w_size = 1<<10;

extern "C" {
  /*** Initialization ***/

  void initializeMPFR_GSL();

  /*** Utility functions ***/

  double logExp(double x, double xmax = 50);
  double logExp_mpfr(double x, double xmax = 50);
}

