#include "expansion.h"

double analytic_prf(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  double z0{fluct_pp_b(a, E, mr_ep, mr_hp, mu_e, mu_h)};

  if (std::isnan(z0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return 1 / (std::exp(z0 / (4 * M_PI)) - 1);
}

double analytic_pr(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  if (a < 0) { return std::numeric_limits<double>::quiet_NaN(); }

  double z0{0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - mu_e - mu_h - 0.25 * a * a};

  return 1 / (std::exp(z0 / (4 * M_PI)) - 1);
}

inline double fluct_i_fl(double a, double b) {
  return std::log((a + b) / (a - b)) / b;
}

double fluct_i_f(double x, void * params) {
  double * params_arr = (double*)params;
  double z{params_arr[0]};
  double E{params_arr[1]};
  double mr_ep{params_arr[2]};
  double mr_hp{params_arr[3]};
  double mu_e{params_arr[4]};
  double mu_h{params_arr[5]};

  constexpr double sqrt_val{-0.1767766953}; //1 / (4 * std::sqrt(2))

  if (E < 1e-7) {
    return 2 * std::sqrt(x) / (z + mu_e + mu_h - x) * ( 1 / (std::exp((mr_hp * x - mu_h) / (4 * M_PI)) + 1) + 1 / (std::exp((mr_ep * x - mu_e) / (4 * M_PI)) + 1)) * sqrt_val;
  }

  double a_e{z + mu_e + mu_h - mr_ep * E - x};
  double a_h{z + mu_e + mu_h - mr_hp * E - x};

  double b_e{2 * mr_ep * std::sqrt(E * x)};
  double b_h{2 * mr_hp * std::sqrt(E * x)};

  return std::sqrt(x) * (fluct_i_fl(a_e, b_e) / (std::exp((mr_hp * x - mu_h) / (4 * M_PI)) + 1) + fluct_i_fl(a_h, b_h) / (std::exp((mr_ep * x - mu_e) / (4 * M_PI)) + 1)) * sqrt_val;
}

double fluct_i_z1_f(double x, void * params) {
  double * params_arr = (double*)params;
  double E{params_arr[0]};
  double mr_ep{params_arr[1]};
  double mr_hp{params_arr[2]};
  double mu_e{params_arr[3]};
  double mu_h{params_arr[4]};

  constexpr double sqrt_val{-0.1767766953}; //1 / (4 * std::sqrt(2))

  if (E < 1e-7) {
    return -2 / std::sqrt(x) * ( 1 / (std::exp((mr_hp * x - mu_h) / (4 * M_PI)) + 1) + 1 / (std::exp((mr_ep * x - mu_e) / (4 * M_PI)) + 1)) * sqrt_val;
  }

  double a_e{0.25 * (1 - std::pow(mr_hp-mr_ep, 2) - 4 * mr_ep) * E - x};
  double a_h{0.25 * (1 - std::pow(mr_hp-mr_ep, 2) - 4 * mr_hp) * E - x};

  double b_e{2 * mr_ep * std::sqrt(E * x)};
  double b_h{2 * mr_hp * std::sqrt(E * x)};

  return std::sqrt(x) * (fluct_i_fl(a_e, b_e) / (std::exp((mr_hp * x - mu_h) / (4 * M_PI)) + 1) + fluct_i_fl(a_h, b_h) / (std::exp((mr_ep * x - mu_e) / (4 * M_PI)) + 1)) * sqrt_val;
}

double fluct_i_z1(double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  constexpr uint32_t n_int{2};
  constexpr double x_max{1e-1};
  double result[n_int] = {0}, error[n_int] = {0};
  double params_arr[] = { E, mr_ep, mr_hp, mu_e, mu_h };

  gsl_function integrand;
  integrand.function = &fluct_i_z1_f;
  integrand.params = params_arr;

  constexpr uint32_t local_ws_size{1<<6};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qags(&integrand, 0, x_max, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_qagiu(&integrand, x_max, 0, std::max(global_eps, error[0]), local_ws_size, ws, result + 1, error + 1);
  gsl_integration_workspace_free(ws);

  //printf("%.3f, %.10f, %.3e\n", E, result[0] + result[1], std::max(error[0], error[1]));

  return result[0] + result[1];
}

double fluct_i(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  constexpr uint32_t n_int{1};
  constexpr double x_max{0};
  double result[n_int] = {0}, error[n_int] = {0};
  double params_arr[] = { z, E, mr_ep, mr_hp, mu_e, mu_h };

  gsl_function integrand;
  integrand.function = &fluct_i_f;
  integrand.params = params_arr;

  constexpr uint32_t local_ws_size{(1<<5)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, x_max, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  //printf("%.3f, %.10f, %.3e\n", z, result[0],  error[0]);

  return result[0];
}

double fluct_i_dfdz_n(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Derivative of the expansion.
  */
  //double z1{0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - mu_e - mu_h};

  if (z > 0) {
    return derivative_b3<1>(&fluct_i, z, z * 1e-3, E, mr_ep, mr_hp, mu_e, mu_h)[0];
  } else {
    return derivative_f3<1>(&fluct_i, z, z * 1e-3, E, mr_ep, mr_hp, mu_e, mu_h)[0];
  }

  //printf("%.3f, %.3f, %3e\n", z, r, err);
}

double fluct_t(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, double a) {
  return 0.5 * M_PI * a - M_PI * std::sqrt(0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - z - mu_e - mu_h) + fluct_i(z, E, mr_ep, mr_hp, mu_e, mu_h);
}

double fluct_t_z1(double E, double mr_ep, double mr_hp, double mu_e, double mu_h, double a) {
  return 0.5 * M_PI * a + fluct_i_z1(E, mr_ep, mr_hp, mu_e, mu_h);
}

double fluct_t_dtdz(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  return 0.5 * M_PI / std::sqrt(0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - z - mu_e - mu_h) + fluct_i_dfdz_n(z, E, mr_ep, mr_hp, mu_e, mu_h);
}

std::complex<double> fluct_t_c(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, double a) {
  return std::complex<double>(1, 2);
}

double fluct_pp_f(double z, void * params) {
  double * params_arr = (double*)params;
  double E = params_arr[0];
  double mr_ep = params_arr[1];
  double mr_hp = params_arr[2];
  double mu_e = params_arr[3];
  double mu_h = params_arr[4];
  double a = params_arr[5];

  return fluct_t(z, E, mr_ep, mr_hp, mu_e, mu_h, a);
}

double fluct_pp_df(double z, void * params) {
  double * params_arr = (double*)params;
  double E = params_arr[0];
  double mr_ep = params_arr[1];
  double mr_hp = params_arr[2];
  double mu_e = params_arr[3];
  double mu_h = params_arr[4];

  return fluct_t_dtdz(z, E, mr_ep, mr_hp, mu_e, mu_h);
}

void fluct_pp_fdf(double z, void * params, double * f, double * df) {
  f[0] = fluct_pp_f(z, params);
  df[0] = fluct_pp_df(z, params);
}

double fluct_pp(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Computes the value z0 that satisfies the following:

    pi / (2 * a) + pi * sqrt((1-(mr_p / mr_m)^2)*E/4 - z0 - mu_e - mu_h) + fluct_es_f(z0, ...) == 0
  */
  double params_arr[] = {E, mr_ep, mr_hp, mu_e, mu_h, a};

  double z1{0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - mu_e - mu_h};

  double T_at_z1{fluct_t_z1(E, mr_ep, mr_hp, mu_e, mu_h, a)};

  if (T_at_z1 < 0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if (T_at_z1 == 0) {
    return z1;
  }

  double z{z1 - 3 / 16 * std::pow(a - fluct_ac_E(E, mr_ep, mr_hp, mu_e, mu_h), 2)}, z0;

  //printf("first: %.2f, %.2f, %.10f, %.10f\n", E, a, T_at_z1, z);

  gsl_function_fdf funct;
  funct.f = &fluct_pp_f;
  funct.df = &fluct_pp_df;
  funct.fdf = &fluct_pp_fdf;
  funct.params = params_arr;

  const gsl_root_fdfsolver_type * T = gsl_root_fdfsolver_steffenson;
  gsl_root_fdfsolver * s = gsl_root_fdfsolver_alloc(T);

  gsl_root_fdfsolver_set(s, &funct, z);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < 16; iter++) {
    status = gsl_root_fdfsolver_iterate(s);
    z0 = z;

    z = gsl_root_fdfsolver_root(s);
    status = gsl_root_test_delta(z, z0, 0, global_eps);

    //printf("iter: %.2f, %.2f, %.10f, %.10f, %d\n", E, a, z0, z, status);
  }

  //printf("result: %.2f, %.2f, %.10f, %.10f, %.10f\n", E, a, z, fluct_pp_f(z, params_arr), fluct_pp_df(z, params_arr));

  gsl_root_fdfsolver_free(s);
  return z;
}

double fluct_pp_b(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Computes the value z0 that satisfies the following:

    pi / (2 * a) + pi * sqrt((1-(mr_p / mr_m)^2)*E/4 - z0 - mu_e - mu_h) + fluct_es_f(z0, ...) == 0

    Brent version.
  */
  double params_arr[] = {E, mr_ep, mr_hp, mu_e, mu_h, a};

  double z1{0.25 * ( 1 - std::pow(mr_hp - mr_ep, 2)) * E - mu_e - mu_h};

  double T_at_z1{fluct_t_z1(E, mr_ep, mr_hp, mu_e, mu_h, a)};

  if (T_at_z1 < 0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if (T_at_z1 == 0) {
    return z1;
  }

  double z{z1 - 0.25 * std::pow(a - fluct_ac_E(E, mr_ep, mr_hp, mu_e, mu_h), 2)};
  double z_min{z}, z_max{0.5 * (z1 + z)};
  //printf("first: %.2f, %.10f, %.3f, %.3f\n", E, a, fluct_pp_f(z_max, params_arr), fluct_pp_f(z_min, params_arr));

  gsl_function funct;
  funct.function = &fluct_pp_f;
  funct.params = params_arr;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < 16; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);

    //printf("iter: %.2f, %.2f, %.10f, %.10f, %d\n", E, a, z0, z, status);
  }

  //printf("result: %.2f, %.2f, %.10f, %.10f, %.10f\n", E, a, z, fluct_pp_f(z, params_arr), fluct_pp_df(z, params_arr));

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pp0_f(double E, void * params) {
  double * params_arr = (double*)params;
  double a = params_arr[0];
  double mr_ep = params_arr[1];
  double mr_hp = params_arr[2];
  double mu_e = params_arr[3];
  double mu_h = params_arr[4];
  return fluct_pp_b(a, E, mr_ep, mr_hp, mu_e, mu_h);
}

double fluct_pp0(double a, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Computes the energy E at which z0(E, a) == 0.
    Brent version.
  */
  double params_arr[] = {a, mr_ep, mr_hp, mu_e, mu_h};

  double ac{fluct_ac(mr_ep, mr_hp, mu_e, mu_h)};
  double z, z_min{0};
  double z_max{(4 * std::abs(mu_e + mu_h) + std::pow(a - ac, 2)) / (1 - std::pow(mr_ep - mr_hp, 2)) + 1};

  double f_min{fluct_pp0_f(z_min, params_arr)};

  if (f_min > 0 || std::isnan(f_min)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  //printf("first: %.3f, %.3f, %.3f, %.3f\n", a, z_max, fluct_pp0_f(z_min, params_arr), fluct_pp0_f(z_max, params_arr));

  gsl_function funct;
  funct.function = &fluct_pp0_f;
  funct.params = params_arr;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < 16; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);

    //printf("iter: %.2f, %.2f, %.10f, %.10f, %d\n", E, a, z0, z, status);
  }

  //printf("result: %.2f, %.2f, %.10f, %.10f, %.10f\n", E, a, z, fluct_pp_f(z, params_arr), fluct_pp_df(z, params_arr));

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pp0c_f(double a, void * params) {
  double * params_arr = (double*)params;
  double mr_ep = params_arr[0];
  double mr_hp = params_arr[1];
  double mu_e = params_arr[2];
  double mu_h = params_arr[3];
  return fluct_pp_b(a, 0, mr_ep, mr_hp, mu_e, mu_h);
}

double fluct_pp0c(double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Computes the energy E at which z0(E, a) == 0.
    Brent version.
  */
  double params_arr[] = {mr_ep, mr_hp, mu_e, mu_h};

  if (mu_e + mu_h > 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double ac{fluct_ac(mr_ep, mr_hp, mu_e, mu_h)};
  double z, z_max{2 * std::sqrt(-(mu_e + mu_h))};
  double z_min{z_max + ac};

  //printf("first: %.3f, %.3f, %.3f, %.3f\n", a, z_max, fluct_pp0_f(z_min, params_arr), fluct_pp0_f(z_max, params_arr));

  gsl_function funct;
  funct.function = &fluct_pp0c_f;
  funct.params = params_arr;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < 16; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);

    //printf("iter: %.2f, %.2f, %.10f, %.10f, %d\n", E, a, z0, z, status);
  }

  //printf("result: %.2f, %.2f, %.10f, %.10f, %.10f\n", E, a, z, fluct_pp_f(z, params_arr), fluct_pp_df(z, params_arr));

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pr(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  double z0{fluct_pp_b(a, E, mr_ep, mr_hp, mu_e, mu_h)};

  if (std::isnan(z0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double z1{0.25 * (1 - std::pow(mr_hp - mr_ep, 2)) * E - mu_e - mu_h};
  double sqrt_val{2 * std::sqrt(z1 - z0) / M_PI};

  return 1 / ( (std::exp(z0 / (4 * M_PI)) - 1) * (1 + sqrt_val * fluct_i_dfdz_n(z0, E, mr_ep, mr_hp, mu_e, mu_h)) );
}

double fluct_pmi_f(double E, void * params) {
  double * params_arr = (double*)params;
  double a{params_arr[0]};
  double mr_ep{params_arr[1]};
  double mr_hp{params_arr[2]};
  double mu_e{params_arr[3]};
  double mu_h{params_arr[4]};

  if (fluct_ac_E(E, mr_ep, mr_hp, mu_e, mu_h) > a) {
    return 0;
  }

  double pr{fluct_pr(a, E, mr_ep, mr_hp, mu_e, mu_h)};
  //double pr{analytic_prf(a, E, mr_ep, mr_hp, mu_e, mu_h)};

  if (std::isnan(pr)) {
    return 0;
  }

  return std::sqrt(E) * pr;
}

double fluct_pmi_nc(double a, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Assuming a < ac_max, therefore no singularities in the integrand.
  */
  constexpr double prefactor{1 / (16 * M_PI)};
  double Emax{fluct_Ec_a(a, mr_ep, mr_hp, mu_e, mu_h)};

  if (std::isnan(Emax)) {
    // a < a_c
    return 0;
  }

  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};
  double params_arr[] = { a, mr_ep, mr_hp, mu_e, mu_h };

  gsl_function integrand;
  integrand.function = &fluct_pmi_f;
  integrand.params = params_arr;

  if (std::isinf(Emax)) {
    constexpr uint32_t local_ws_size{1<<3};
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_workspace_free(ws);
  } else {
    if (Emax > 1e3) {
      constexpr uint32_t local_ws_size{1<<3};
      gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
      gsl_integration_qag(&integrand, 0, Emax, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result, error);
      gsl_integration_workspace_free(ws);
    } else {
      size_t neval[1] = {0};
      gsl_integration_qng(&integrand, 0, Emax, 0, global_eps, result, error, neval);
    }
  }

  //printf("%.3f, %.10f, %.3e\n", a, result[0] * prefactor,  error[0]);

  return prefactor * (result[0]);
}

double fluct_pmi(double a, double ac_max, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  constexpr double prefactor{- 1 / (16 * M_PI)};
  double Emax{fluct_Ec_a(a, mr_ep, mr_hp, mu_e, mu_h)};

  if (std::isnan(Emax)) {
    // a < a_c
    return 0;
  }

  /*
    The integrand is singular when z0(E) == 0,
    so we will compute z0(0), and if it is positive
    we need to compute the integral considering the
    singularity.
  */

  //double z0_E0{fluct_pr(a, 0, mr_ep, mr_hp, mu_e, mu_h)};

  constexpr uint32_t n_int{2};
  double result[n_int] = {0}, error[n_int] = {0};
  double params_arr[] = { a, mr_ep, mr_hp, mu_e, mu_h };

  gsl_function integrand;
  integrand.function = &fluct_pmi_f;
  integrand.params = params_arr;

  if (a < ac_max) {
    if (std::isinf(Emax)) {
      constexpr uint32_t local_ws_size{1<<3};
      gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
      gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
      gsl_integration_workspace_free(ws);
    } else {
      //gsl_integration_qag(&integrand, 0, Emax, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result, error);
      size_t neval[1] = {0};
      gsl_integration_qng(&integrand, 0, Emax, 0, global_eps, result, error, neval);
    }
  } else {
    double Epole{fluct_pp0(a, mr_ep, mr_hp, mu_e, mu_h)};
    constexpr uint32_t local_ws_size{1<<4};
    //size_t neval[1] = {0};
    double pts[] = {0, Epole, 2 * Epole};

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    //gsl_integration_qags(&integrand, pts[0], pts[2], 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_qagp(&integrand, pts, 3, 0, global_eps, local_ws_size, ws, result, error);
    //gsl_integration_qng(&integrand, 0, 2 * Epole, 0, global_eps, result, error, neval);

    gsl_integration_qagiu(&integrand, pts[2], 0, global_eps, local_ws_size, ws, result + 1, error + 1);
    gsl_integration_workspace_free(ws);
  }

  //printf("%.3f, %.10f, %.3e\n", a, result[0] * prefactor,  error[0]);

  return prefactor * (result[0] + result[1]);
}

double fluct_ac(double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Critical scattering length a_c at E = 0, meaning that for any a < a_c there
    won't be any excitons.
  */
  return - 2 / M_PI * fluct_i_z1(0, mr_ep, mr_hp, mu_e, mu_h);
  //return -(std::sqrt(2 / mr_ep) * polylogExpM(0.5, mu_e / (4 * M_PI)) + std::sqrt(2 / mr_hp) * polylogExpM(0.5, mu_h / (4 * M_PI)));
}

double fluct_ac_E(double E, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Critical scattering length a_c(E), which always satisfies a_c(E) > a_c(0) = a_c.
  if (E == 0) {
    return fluct_ac(mr_ep, mr_hp, mu_e, mu_h);
  }
  */
  return - 2 / M_PI * fluct_i_z1(E, mr_ep, mr_hp, mu_e, mu_h);
}

double fluct_Ec_a_f(double E, void * params) {
  double * params_arr = (double*)params;
  double a{params_arr[0]};
  double mr_ep{params_arr[1]};
  double mr_hp{params_arr[2]};
  double mu_e{params_arr[3]};
  double mu_h{params_arr[4]};

  return fluct_ac_E(E, mr_ep, mr_hp, mu_e, mu_h) - a;
}

double fluct_Ec_a(double a, double mr_ep, double mr_hp, double mu_e, double mu_h) {
  /*
    Given some scattering length a, E_c(a) is the maximum kinetic energy
    that will contain excitons. It satisfies E_c(a_c) = 0.
  */
  double params_arr[] = {a, mr_ep, mr_hp, mu_e, mu_h};

  double ac{fluct_ac(mr_ep, mr_hp, mu_e, mu_h)};

  if (a < ac) {
    // There is no solution for a < a_c.
    return std::numeric_limits<double>::quiet_NaN();
  } else if (a >= 0) {
    // For a >= 0 there is no maximum energy.
    return std::numeric_limits<double>::infinity();
  } else if (a == ac) {
    // If a is exactly a_c, then E = 0.
    return 0;
  }

  double z, z_min{0}, z_max{1};

  /*
    Exponential sweep to find bounds.
  */

  for (uint32_t i = 0; i < 32; z_max *= 8) {
    // As E increases, a_c(E) -> 0, so at some point a_c(E) < a.
    if (fluct_Ec_a_f(z_max, params_arr) > 0) {
      break;
    }
    z_min = z_max;
  }

  //printf("first: %.2f, %.2e, %.2e, %.10f, %.10f\n", a, z_min, z_max, fluct_Ec_a_f(z_min, params_arr), fluct_Ec_a_f(z_max, params_arr));

  gsl_function funct;
  funct.function = &fluct_Ec_a_f;
  funct.params = params_arr;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < 16; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);

    //printf("iter: %.2f, %.2f, %.10f, %.10f, %d\n", E, a, z0, z, status);
  }

  //printf("result: %.2f, %.10f, %.10f, %.10f\n", a, z, fluct_Ec_a_f(z, params_arr), fluct_Ec_a_df(z, params_arr));

  gsl_root_fsolver_free(s);
  return z;
}

