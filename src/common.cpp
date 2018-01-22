#include "common.h"

/*** Initialization ***/

void initializeMPFR_GSL() {
  mpfr_set_default_prec(prec);
  gsl_set_error_handler_off();
}


/*** Utility functions ***/

double logExp(double x, double xmax) {
  // log( 1 + exp( x ) )
  double d_y;

  if (x < xmax) {
    // Approximate log(exp(x) + 1) ~ x when x > xmax
    d_y = log(1 + exp(x));
  } else {
    d_y = x;
  }

  return d_y;
}

double logExp_mpfr(double x, double xmax) {
  // log( 1 + exp( x ) )
  double d_y;

  if (x < xmax) {
    // Approximate log(exp(x) + 1) ~ x when x > xmax
    mpfr_t y;
    mpfr_init_set_d(y, x, MPFR_RNDN);
    mpfr_exp(y, y, MPFR_RNDN);
    mpfr_add_ui(y, y, 1, MPFR_RNDN);
    mpfr_log(y, y, MPFR_RNDN);

    d_y = mpfr_get_d(y, MPFR_RNDN);

    mpfr_clear(y);
  } else {
    d_y = x;
  }

  return d_y;
}

double polylogExp(double s, double z) {
  double r;

  if (z < 0) {
    arb_t arb_x, arb_s, arb_z;

    arb_init(arb_x);
    arb_init(arb_s);
    arb_init(arb_z);

    arb_set_d(arb_s, s);
    arb_set_d(arb_z, z);

    arb_exp(arb_z, arb_z, prec);

    arb_polylog(arb_x, arb_s, arb_z, prec);

    r = arf_get_d(arb_midref(arb_x), ARF_RND_NEAR);

    arb_clear(arb_x);
    arb_clear(arb_s);
    arb_clear(arb_z);
  } else {
    acb_t acb_x, acb_s, acb_z;

    acb_init(acb_x);
    acb_init(acb_s);
    acb_init(acb_z);

    acb_set_d(acb_s, s);
    acb_set_d(acb_z, z);

    acb_exp(acb_z, acb_z, prec);

    acb_polylog(acb_x, acb_s, acb_z, prec);

    r = arf_get_d(arb_midref(acb_realref(acb_x)), ARF_RND_NEAR);

    acb_clear(acb_x);
    acb_clear(acb_s);
    acb_clear(acb_z);
  }

  return r;
}

double polylogExpM(double s, double z) {
  arb_t arb_x, arb_s, arb_z;

  arb_init(arb_x);
  arb_init(arb_s);
  arb_init(arb_z);

  arb_set_d(arb_s, s);
  arb_set_d(arb_z, z);

  arb_exp(arb_z, arb_z, prec);
  arb_neg(arb_z, arb_z);

  arb_polylog(arb_x, arb_s, arb_z, prec);

  double r = arf_get_d(arb_midref(arb_x), ARF_RND_NEAR);

  arb_clear(arb_x);
  arb_clear(arb_s);
  arb_clear(arb_z);

  return r;
}

double erf_fk(double x, double y, uint32_t k) {
  return 2 * x * ( 1 - std::cos(2 * x * y) * std::cosh(k * y) + k * std::sin(2 * x * y) * std::sinh(k * y) );
}

double erf_gk(double x, double y, uint32_t k) {
  return 2 * x * std::sin(2 * x * y) * std::cosh(k * y) + k * std::cos(2 * x * y) * std::sinh(k * y);
}

double erf_sterm_r(double x, double y, double k) {
  return std::exp(- 0.25 * k*k) * erf_fk(x, y, k) / ( k*k + 4 * x*x);
}

double erf_sterm_i(double x, double y, double k) {
  return std::exp(- 0.25 * k*k) * erf_gk(x,y,k) / ( k*k + 4 * x*x);
}

double erf_r(double x, double y, uint32_t n, double eps) {
  double constant_add = std::erf(x) + std::exp(-x*x) / ( 2 * M_PI * x) * ( 1 - std::cos(2 * x * y) );
  double constant_prod = 2 * std::exp(-x*x) / M_PI;
  double val_prev = 0, val_curr, err = 1;

  val_curr = constant_prod * erf_sterm_r(x, y, 1);

  for(uint32_t k = 2; k < n && err > eps; k++) {
    val_prev += val_curr;
    val_curr = constant_prod * erf_sterm_r(x, y, k);

    err = std::abs(val_curr - val_prev);
  }

  return constant_add + val_curr + val_prev;
}

double erf_i(double x, double y, uint32_t n, double eps) {
  double constant_add = std::exp(-x*x) / ( 2 * M_PI * x) * std::sin(2 * x * y);
  double constant_prod = 2 * std::exp(-x*x) / M_PI;
  double val_prev = 0, val_curr, err = 1;

  val_curr = constant_prod * erf_sterm_i(x, y, 1);

  for(uint32_t k = 2; k < n && err > eps; k++) {
    val_prev += val_curr;
    val_curr = constant_prod * erf_sterm_i(x, y, k);

    err = std::abs(val_curr - val_prev);
  }

  return constant_add + val_curr + val_prev;
}
