#include "common.h"

/*** Utility functions ***/

double logExp(double x, double xmax) {
  // log( 1 + exp( x ) )
  double d_y;

  if (x < xmax) {
    // Approximate log(exp(x) + 1) ~ x when x > xmax

    if constexpr (use_mpfr) {
      mpfr_t y;
      mpfr_init_set_d(y, x, MPFR_RNDN);
      mpfr_exp(y, y, MPFR_RNDN);
      mpfr_add_ui(y, y, 1, MPFR_RNDN);
      mpfr_log(y, y, MPFR_RNDN);

      d_y = mpfr_get_d(y, MPFR_RNDN);

      mpfr_clear(y);
    } else {
      d_y = log(1 + exp(x));
    }
  } else {
    d_y = x;
  }

  return d_y;
}

