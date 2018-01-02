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

