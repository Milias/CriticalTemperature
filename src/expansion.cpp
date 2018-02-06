#include "expansion.h"

double fluct_e_tf(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}

    TODO: check this, there seems to be some difference compared with Mathematica's result.
    This is related to the zeroth order term. Solved?
  */

  double arg_e{mr_hp * (mr_ep * E - mu_e - mu_h - z) / (4 * M_PI)};
  double arg_h{mr_ep * (mr_hp * E - mu_e - mu_h - z) / (4 * M_PI)};

  //printf("arg: %d, %f, %f\n", n, arg_e, arg_h);

  double prefactor_e{std::pow(mr_ep * mr_ep * E * mr_hp / M_PI, n) * std::exp(mu_h / (4 * M_PI)) / ( std::sqrt(mr_hp / (4 * M_PI)) * (2 * n + 1))};
  double prefactor_h{std::pow(mr_hp * mr_hp * E * mr_ep / M_PI, n) * std::exp(mu_e / (4 * M_PI)) / ( std::sqrt(mr_ep / (4 * M_PI)) * (2 * n + 1))};

  //printf("prefactors: %d, %.3f, %.10e, %.10e\n", n, E, prefactor_e, prefactor_h);

  double val1_e{gsl_sf_beta(n - 0.5, n + 1.5) * std::pow(arg_e, 0.5 - n) * gsl_sf_hyperg_1F1(n + 1.5, 1.5 - n, arg_e)};
  double val1_h{gsl_sf_beta(n - 0.5, n + 1.5) * std::pow(arg_h, 0.5 - n) * gsl_sf_hyperg_1F1(n + 1.5, 1.5 - n, arg_h)};

  //printf("val1: %d, %.3f, %.10e, %.10e\n", n, E, val1_e, val1_h);

  double val2_e{gsl_sf_gamma(0.5 - n) * gsl_sf_hyperg_1F1(2 * n + 1, n + 0.5, arg_e)};
  double val2_h{gsl_sf_gamma(0.5 - n) * gsl_sf_hyperg_1F1(2 * n + 1, n + 0.5, arg_h)};

  //printf("val2: %d, %.3f, %.10e, %.10e\n", n, E, val2_e, val2_h);

  double r{prefactor_e * (val1_e + val2_e) + prefactor_h * (val1_h + val2_h)};

  //printf("result: %d, %f, %f\n", n, E, r);

  return r;
}

double fluct_e_tdf_1e(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.
    Only first term without prefactor.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double arg_e{mr_hp * (mr_ep * E - mu_e - mu_h - z) / (4 * M_PI)};

  double val1_e{(n - 0.5) * gsl_sf_beta(n - 0.5, n + 1.5) * std::pow(arg_e, 0.5 - n) * gsl_sf_hyperg_1F1(n + 1.5, 1.5 - n, arg_e) / (mr_ep * E - mu_e - mu_h - z)};

  return val1_e;
}


double fluct_e_tdf_1h(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.
    Only first term without prefactor.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double arg_h{mr_ep * (mr_hp * E - mu_e - mu_h - z) / (4 * M_PI)};

  double val1_h{(n - 0.5) * gsl_sf_beta(n - 0.5, n + 1.5) * std::pow(arg_h, 0.5 - n) * gsl_sf_hyperg_1F1(n + 1.5, 1.5 - n, arg_h) / (mr_ep * E - mu_e - mu_h - z)};

  return val1_h;
}

double fluct_e_tdf_2e(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.
    Only second term without prefactor.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double arg_e{mr_hp * (mr_ep * E - mu_e - mu_h - z) / (4 * M_PI)};

  double val1_e{mr_hp / (4 * M_PI) * (n + 1.5) / (n - 1.5) * gsl_sf_beta(n - 0.5, n + 1.5) * std::pow(arg_e, 0.5 - n) * gsl_sf_hyperg_1F1(n + 2.5, 2.5 - n, arg_e)};

  return val1_e;
}


double fluct_e_tdf_2h(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.
    Only second term without prefactor.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double arg_h{mr_ep * (mr_hp * E - mu_e - mu_h - z) / (4 * M_PI)};

  double val1_h{mr_ep / (4 * M_PI) * (n + 1.5) / (n - 1.5) * gsl_sf_beta(n - 0.5, n + 1.5) * std::pow(arg_h, 0.5 - n) * gsl_sf_hyperg_1F1(n + 2.5, 2.5 - n, arg_h)};

  return val1_h;
}

double fluct_e_tdf_3e(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.
    Only third term without prefactor.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double arg_e{mr_hp * (mr_ep * E - mu_e - mu_h - z) / (4 * M_PI)};

  double val2_e{-mr_hp / (4 * M_PI) * (2 * n + 1) / (n + 0.5) * gsl_sf_gamma(0.5 - n) * gsl_sf_hyperg_1F1(2 * n + 2, n + 1.5, arg_e)};

  return val2_e;
}


double fluct_e_tdf_3h(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.
    Only third term without prefactor.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double arg_h{mr_ep * (mr_hp * E - mu_e - mu_h - z) / (4 * M_PI)};

  double val2_h{-mr_ep / (4 * M_PI) * (2 * n + 1) / (n + 0.5) * gsl_sf_gamma(0.5 - n) * gsl_sf_hyperg_1F1(2 * n + 2, n + 1.5, arg_h)};

  return val2_h;
}

double fluct_e_tdf(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n) {
  /*
    Computes the n-th term of the expansion in the fluctuations contribution derivative w.r.t. z.

    E = \epsilon_{k, +}
    mr_ep, mr_hp = \frac{m_+}{m_{e,h}}
  */

  double prefactor_e{std::pow(mr_ep * mr_ep * E * mr_hp / M_PI, n) * std::exp(mu_h / (4 * M_PI)) / ( std::sqrt(mr_hp / (4 * M_PI)) * (2 * n + 1))};
  double prefactor_h{std::pow(mr_hp * mr_hp * E * mr_ep / M_PI, n) * std::exp(mu_e / (4 * M_PI)) / ( std::sqrt(mr_ep / (4 * M_PI)) * (2 * n + 1))};

  return prefactor_e * (
    fluct_e_tdf_1e(z, E, mr_ep, mr_hp, mu_e, mu_h, n) +
    fluct_e_tdf_2e(z, E, mr_ep, mr_hp, mu_e, mu_h, n) +
    fluct_e_tdf_3e(z, E, mr_ep, mr_hp, mu_e, mu_h, n)
    ) + prefactor_h * (
    fluct_e_tdf_1h(z, E, mr_ep, mr_hp, mu_e, mu_h, n) +
    fluct_e_tdf_2h(z, E, mr_ep, mr_hp, mu_e, mu_h, n) +
    fluct_e_tdf_3h(z, E, mr_ep, mr_hp, mu_e, mu_h, n)
  );
}

double fluct_es_f(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  constexpr uint32_t nmax = 8;

  double terms[nmax] = {0};
  double sum_accel, err;

  gsl_sum_levin_u_workspace * ws = gsl_sum_levin_u_alloc(nmax);

  for (uint32_t i = 0; i < nmax; i++) {
    terms[i] = fluct_e_tf(z, E, mr_ep, mr_hp, mu_e, mu_h, i);
    //printf("%d, %f, %e\n", i, E, terms[i]);
  }

  gsl_sum_levin_u_accel(terms, nmax, ws, &sum_accel, &err);

  //sum_accel = ws->sum_plain;

  //printf("sum: %.3f, %.10f (%.10f), %.10f\n", E, sum_accel, ws->sum_plain, err);

  gsl_sum_levin_u_free(ws);

  return sum_accel;
}

double fluct_es_df(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Expansion of the derivative.
  */
  constexpr uint32_t nmax = 16;

  double terms[nmax] = {0};
  double sum_accel, err;

  gsl_sum_levin_u_workspace * ws = gsl_sum_levin_u_alloc(nmax);

  for (uint32_t i = 0; i < nmax; i++) {
    terms[i] = fluct_e_tdf(z, E, mr_ep, mr_hp, mu_e, mu_h, i);
  }

  gsl_sum_levin_u_accel(terms, nmax, ws, &sum_accel, &err);

  //printf("sum: %.3f, %.10f (%.10f), %.10f\n", E, sum_accel, ws->sum_plain, err);

  gsl_sum_levin_u_free(ws);

  return sum_accel;
}

double fluct_es_dfn(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Derivative of the expansion.
  */
  return derivative_b3<1>(&fluct_es_f, z, z * 1e-7, E, mr_ep, mr_hp, mu_e, mu_h)[0];
}

double fluct_pf_f(double z, void * params) {
  double * params_arr = (double*)params;
  double E = params_arr[0];
  double mr_ep = params_arr[1];
  double mr_hp = params_arr[2];
  double mu_e = params_arr[3];
  double mu_h = params_arr[4];
  double a = params_arr[5];

  return 0.5 * M_PI * a - M_PI * std::sqrt(0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - z - mu_e - mu_h) + fluct_es_f(z, E, mr_ep, mr_hp, mu_e, mu_h);
}

double fluct_pf_df(double z, void * params) {
  double * params_arr = (double*)params;
  double E = params_arr[0];
  double mr_ep = params_arr[1];
  double mr_hp = params_arr[2];
  double mu_e = params_arr[3];
  double mu_h = params_arr[4];

  return 0.5 * M_PI / std::sqrt(0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - z - mu_e - mu_h) + fluct_es_dfn(z, E, mr_ep, mr_hp, mu_e, mu_h);
}

void fluct_pf_fdf(double z, void * params, double * f, double * df) {
  f[0] = fluct_pf_f(z, params);
  df[0] = fluct_pf_df(z, params);
}

double fluct_pf(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Computes the value z0 that satisfies the following:

    pi / (2 * a) + pi * sqrt((1-(mr_p / mr_m)^2)*E/4 - z0 - mu_e - mu_h) + fluct_es_f(z0, ...) == 0
  */
  double params_arr[] = {E, mr_ep, mr_hp, mu_e, mu_h, a};

  double z1{0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - mu_e - mu_h - 1e-3};

  double T_at_z1{fluct_pf_f(z1, params_arr)};

  if (T_at_z1 < 0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if (T_at_z1 == 0) {
    return z1;
  }

  double z{a > 0 ? z1 - 0.25 * a * a : z1 - 1e-10}, z0;

  //printf("first: %.2f, %.2f, %.10f, %.10f\n", E, a, T_at_z1, z);

  gsl_function_fdf funct;
  funct.f = &fluct_pf_f;
  funct.df = &fluct_pf_df;
  funct.fdf = &fluct_pf_fdf;
  funct.params = params_arr;

  const gsl_root_fdfsolver_type * T = gsl_root_fdfsolver_steffenson;
  gsl_root_fdfsolver * s = gsl_root_fdfsolver_alloc(T);

  gsl_root_fdfsolver_set(s, &funct, z);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < 64; iter++) {
    status = gsl_root_fdfsolver_iterate(s);
    z0 = z;

    z = gsl_root_fdfsolver_root(s);
    status = gsl_root_test_delta(z, z0, 0, 1e-10);

    //printf("iter: %.2f, %.2f, %.10f, %.10f, %d\n", E, a, z0, z, status);
  }

  //printf("result: %.2f, %.2f, %.10f, %.10f, %.10f\n", E, a, z, fluct_pf_f(z, params_arr), fluct_pf_df(z, params_arr));

  gsl_root_fdfsolver_free(s);
  return z;
}

