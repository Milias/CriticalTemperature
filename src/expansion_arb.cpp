#include "expansion.h"

double fluct_e_tfr(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}

    In this function we compute it using arbitrary precision.
  */

  double arg_e{mr_hp * (mr_ep * E - mu_e - mu_h - z) / (4 * M_PI)};
  double arg_h{mr_ep * (mr_hp * E - mu_e - mu_h - z) / (4 * M_PI)};

  /* init: arb_arg */
  arb_t arb_arg_e, arb_arg_h;
  arb_init(arb_arg_e);
  arb_init(arb_arg_h);

  arb_set_d(arb_arg_e, arg_e);
  arb_set_d(arb_arg_h, arg_h);

  //printf("arg: %d, %f, %f, %f\n", n, E, arf_get_d(arb_midref(arb_arg_e), ARF_RND_NEAR), arf_get_d(arb_midref(arb_arg_h), ARF_RND_NEAR));

  /* init: arb_prefactor */
  arb_t arb_prefactor_e, arb_prefactor_h;
  arb_init(arb_prefactor_e);
  arb_init(arb_prefactor_h);

  arb_set_d(arb_prefactor_e, mr_ep * mr_ep * E * mr_hp / M_PI);
  arb_set_d(arb_prefactor_h, mr_hp * mr_hp * E * mr_ep / M_PI);

  arb_pow_ui(arb_prefactor_e, arb_prefactor_e, n, ARF_RND_NEAR);
  arb_pow_ui(arb_prefactor_h, arb_prefactor_h, n, ARF_RND_NEAR);

  /* init: arb_exp */
  arb_t arb_exp_e, arb_exp_h;
  arb_init(arb_exp_e);
  arb_init(arb_exp_h);

  arb_set_d(arb_exp_e, mu_h / (4 * M_PI));
  arb_set_d(arb_exp_h, mu_e / (4 * M_PI));

  arb_exp(arb_exp_e, arb_exp_e, ARF_RND_NEAR);
  arb_exp(arb_exp_h, arb_exp_h, ARF_RND_NEAR);

  arb_mul(arb_prefactor_e, arb_exp_e, arb_prefactor_e, ARF_RND_NEAR);
  arb_mul(arb_prefactor_h, arb_exp_h, arb_prefactor_h, ARF_RND_NEAR);

  arb_clear(arb_exp_e);
  arb_clear(arb_exp_h);
  /* clear: arb_exp */

  arb_div_ui(arb_prefactor_e, arb_prefactor_e, 2 * n + 1, ARF_RND_NEAR);
  arb_div_ui(arb_prefactor_h, arb_prefactor_h, 2 * n + 1, ARF_RND_NEAR);

  /* init: arb_sqrt */
  arb_t arb_sqrt_e, arb_sqrt_h;
  arb_init(arb_sqrt_e);
  arb_init(arb_sqrt_h);

  arb_set_d(arb_sqrt_e, mr_hp / (4 * M_PI));
  arb_set_d(arb_sqrt_h, mr_ep / (4 * M_PI));

  arb_sqrt(arb_sqrt_e, arb_sqrt_e, ARF_RND_NEAR);
  arb_sqrt(arb_sqrt_h, arb_sqrt_h, ARF_RND_NEAR);

  arb_div(arb_prefactor_e, arb_prefactor_e, arb_sqrt_e, ARF_RND_NEAR);
  arb_div(arb_prefactor_h, arb_prefactor_h, arb_sqrt_h, ARF_RND_NEAR);

  arb_clear(arb_sqrt_e);
  arb_clear(arb_sqrt_h);
  /* clear: arb_sqrt */

  /* init: arb_hyperg, arb_val1 */
  arb_t arb_val1_e, arb_hyperg_e;
  arb_t arb_val1_h, arb_hyperg_h;

  arb_init(arb_val1_e);
  arb_init(arb_val1_h);
  arb_init(arb_hyperg_e);
  arb_init(arb_hyperg_h);

  /* init: arb_hypgeom_a, arb_hypgeom_b */
  arb_t arb_hypgeom_a, arb_hypgeom_b;
  arb_init(arb_hypgeom_a);
  arb_init(arb_hypgeom_b);

  arb_set_d(arb_hypgeom_a, 0.5 - n);

  arb_pow(arb_val1_e, arb_arg_e, arb_hypgeom_a, ARF_RND_NEAR);
  arb_pow(arb_val1_h, arb_arg_h, arb_hypgeom_a, ARF_RND_NEAR);

  arb_set_d(arb_hypgeom_a, n + 1.5);
  arb_set_d(arb_hypgeom_b, 1.5 - n);

  //printf("hypgeom: %d, %f, %f, %f\n", n, E, arf_get_d(arb_midref(arb_hypgeom_a), ARF_RND_NEAR), arf_get_d(arb_midref(arb_hypgeom_b), ARF_RND_NEAR));

  arb_set_d(arb_hyperg_e, gsl_sf_hyperg_1F1(n + 1.5, 1.5 - n, arg_e));
  arb_set_d(arb_hyperg_h, gsl_sf_hyperg_1F1(n + 1.5, 1.5 - n, arg_h));

  //arb_hypgeom_1f1(arb_hyperg_e, arb_hypgeom_a, arb_hypgeom_b, arb_arg_e, 0, ARF_RND_NEAR);
  //arb_hypgeom_1f1(arb_hyperg_h, arb_hypgeom_a, arb_hypgeom_b, arb_arg_h, 0, ARF_RND_NEAR);

  //printf("hyperg: %d, %f, %f, %f\n", n, E, arf_get_d(arb_midref(arb_hyperg_e), ARF_RND_NEAR), arf_get_d(arb_midref(arb_hyperg_h), ARF_RND_NEAR));

  /* init: arb_beta */
  arb_t arb_beta;
  arb_init(arb_beta);

  arb_set_d(arb_beta, gsl_sf_beta(n - 0.5, n + 1.5));

  arb_mul(arb_hyperg_e, arb_hyperg_e, arb_beta, ARF_RND_NEAR);
  arb_mul(arb_hyperg_h, arb_hyperg_h, arb_beta, ARF_RND_NEAR);

  arb_mul(arb_val1_e, arb_val1_e, arb_hyperg_e, ARF_RND_NEAR);
  arb_mul(arb_val1_h, arb_val1_h, arb_hyperg_h, ARF_RND_NEAR);

  //printf("val1: %d, %f, %f, %f\n", n, E, arf_get_d(arb_midref(arb_val1_e), ARF_RND_NEAR), arf_get_d(arb_midref(arb_val1_h), ARF_RND_NEAR));

  /* init: arb_val2 */
  arb_t arb_val2_e, arb_val2_h;
  arb_init(arb_val2_e);
  arb_init(arb_val2_h);

  arb_set_ui(arb_hypgeom_a, 2 * n + 1);
  arb_set_d(arb_hypgeom_b, n + 0.5);

  arb_set_d(arb_hyperg_e, gsl_sf_hyperg_1F1(2 * n + 1, n + 0.5, arg_e));
  arb_set_d(arb_hyperg_h, gsl_sf_hyperg_1F1(2 * n + 1, n + 0.5, arg_h));

  //arb_hypgeom_1f1(arb_hyperg_e, arb_hypgeom_a, arb_hypgeom_b, arb_arg_e, 0, ARF_RND_NEAR);
  //arb_hypgeom_1f1(arb_hyperg_h, arb_hypgeom_a, arb_hypgeom_b, arb_arg_h, 0, ARF_RND_NEAR);

  arb_set_d(arb_beta, gsl_sf_gamma(0.5 - n));

  arb_mul(arb_val2_e, arb_hyperg_e, arb_beta, ARF_RND_NEAR);
  arb_mul(arb_val2_h, arb_hyperg_h, arb_beta, ARF_RND_NEAR);

  arb_clear(arb_arg_e);
  arb_clear(arb_arg_h);
  arb_clear(arb_beta);
  /* clear: arb_beta, arb_arg */

  arb_clear(arb_hyperg_e);
  arb_clear(arb_hyperg_h);
  arb_clear(arb_hypgeom_a);
  arb_clear(arb_hypgeom_b);
  /* clear: arb_hyperg, arb_hypgeom */

  //printf("prefactor: %d, %f, %f, %f\n", n, E, arf_get_d(arb_midref(arb_prefactor_e), ARF_RND_NEAR), arf_get_d(arb_midref(arb_prefactor_h), ARF_RND_NEAR));

  arb_add(arb_val1_e, arb_val1_e, arb_val2_e, ARF_RND_NEAR);
  arb_add(arb_val1_h, arb_val1_h, arb_val2_h, ARF_RND_NEAR);

  arb_mul(arb_prefactor_e, arb_prefactor_e, arb_val1_e, ARF_RND_NEAR);
  arb_mul(arb_prefactor_h, arb_prefactor_h, arb_val1_h, ARF_RND_NEAR);

  arb_add(arb_prefactor_e, arb_prefactor_e, arb_prefactor_h, ARF_RND_NEAR);

  printf("result: %d, %f, %f\n", n, E, arf_get_d(arb_midref(arb_prefactor_e), ARF_RND_NEAR));

  double r{arf_get_d(arb_midref(arb_prefactor_e), ARF_RND_NEAR)};

  arb_clear(arb_val1_e);
  arb_clear(arb_val1_h);
  /* clear: arb_val1 */

  arb_clear(arb_val2_e);
  arb_clear(arb_val2_h);
  /* clear: arb_val2 */

  arb_clear(arb_prefactor_e);
  arb_clear(arb_prefactor_h);
  /* clear: arb_prefactor */

  return r;
}

