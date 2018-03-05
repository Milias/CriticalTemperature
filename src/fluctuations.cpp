#include "fluctuations.h"
#include "fluctuations_utils.h"

double fluct_T_i_fl(double a, double b) {
  return std::log((a + b) / (a - b)) / b;
}

double fluct_T_i_f_E0(double x, fluct_T_i_s * s) {
  /*
   * If E << 1, the logarithm can be simplified,
   * which means only the first contribution is
   * important.
   */

  return 2 * std::sqrt(x)
    / (s->z + s->mu_e + s->mu_h - x)
    * (
      1 / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1) +
      1 / (std::exp((s->sys.m_pe * x - s->mu_e) / (4 * M_PI)) + 1)
    ) * s->c_f;
}

double fluct_T_i_f(double x, void * params) {
  fluct_T_i_s * s{static_cast<fluct_T_i_s*>(params)};

  if (s->E < global_eps) {
    return fluct_T_i_f_E0(x, s);
  }

  double a_e{s->z + s->mu_e + s->mu_h - s->sys.m_pe * s->E - x};
  double a_h{s->z + s->mu_e + s->mu_h - s->sys.m_ph * s->E - x};

  double b_e{2 * s->sys.m_pe * std::sqrt(s->E * x)};
  double b_h{2 * s->sys.m_ph * std::sqrt(s->E * x)};

  return std::sqrt(x) * (
      fluct_T_i_fl(a_e, b_e) /
        (std::exp((s->sys.m_ph * x - s->mu_h) /
        (4 * M_PI)) + 1) +
      fluct_T_i_fl(a_h, b_h) /
        (std::exp((s->sys.m_pe * x - s->mu_e) /
        (4 * M_PI)) + 1)
    ) * s->c_f;
}

double fluct_T_i(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{1};
  constexpr uint32_t local_ws_size{(1<<4)};

  double result[n_int] = {0}, error[n_int] = {0};

  fluct_T_i_s s{z, E, mu_e, mu_h, sys};

  gsl_function integrand;
  integrand.function = &fluct_T_i_f;
  integrand.params = &s;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  return sum_result<n_int>(result);
}

double fluct_T(double z, double E, double a, double mu_e, double mu_h, const system_data & sys) {
  return
    0.5 * M_PI * a -
    M_PI * std::sqrt(0.25 * ( 1 - std::pow(sys.m_ph - sys.m_pe, 2)) * E - z - mu_e - mu_h) +
    fluct_T_i(z, E, mu_e, mu_h, sys);
}

double fluct_T_z1_i_f_E0(double x, fluct_T_i_s * s) {
  /*
   * If E << 1, the logarithm can be simplified,
   * which means only the first contribution is
   * important.
   */

  return - 2 /
    std::sqrt(x) *
    (
      1 / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1) +
      1 / (std::exp((s->sys.m_pe * x - s->mu_e) / (4 * M_PI)) + 1)
    ) * s->c_f;
}

double fluct_T_z1_i_f(double x, void * params) {
  fluct_T_z1_i_s * s{static_cast<fluct_T_z1_i_s*>(params)};

  if (s->E < global_eps) {
    return -2 / std::sqrt(x) * ( 1 / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1) + 1 / (std::exp((s->sys.m_pe * x - s->mu_e) / (4 * M_PI)) + 1)) * s->c_f;
  }

  double a_e{
    0.25 * (
      1 - std::pow(s->sys.m_ph-s->sys.m_pe, 2) - 4 * s->sys.m_pe
    ) * s->E - x
  };
  double a_h{
    0.25 * (
      1 - std::pow(s->sys.m_ph-s->sys.m_pe, 2) - 4 * s->sys.m_ph
    ) * s->E - x
  };

  double b_e{2 * s->sys.m_pe * std::sqrt(s->E * x)};
  double b_h{2 * s->sys.m_ph * std::sqrt(s->E * x)};

  return std::sqrt(x) * (
      fluct_T_i_fl(a_e, b_e) /
        (std::exp((s->sys.m_ph * x - s->mu_h) /
        (4 * M_PI)) + 1) +
      fluct_T_i_fl(a_h, b_h) /
        (std::exp((s->sys.m_pe * x - s->mu_e) /
        (4 * M_PI)) + 1)
    ) * s->c_f;
}

double fluct_T_z1_i(double E, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{2};
  constexpr double x_max{1e-1};
  double result[n_int] = {0}, error[n_int] = {0};

  fluct_T_z1_i_s s{E, mu_e, mu_h, sys};

  gsl_function integrand;
  integrand.function = &fluct_T_z1_i_f;
  integrand.params = &s;

  constexpr uint32_t local_ws_size{(1<<4)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qags(&integrand, 0, x_max, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_qagiu(&integrand, x_max, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  return sum_result<n_int>(result);
}

double fluct_T_z1(double E, double a, double mu_e, double mu_h, const system_data & sys) {
  return
    0.5 * M_PI * a +
    fluct_T_z1_i(E, mu_e, mu_h, sys);
}

double fluct_dT_i_dz(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
  if (z > 0) {
    return derivative_b3(&fluct_T_i, z, z * 1e-6, E, mu_e, mu_h, sys);
  } else {
    return derivative_f3(&fluct_T_i, z, z * 1e-6, E, mu_e, mu_h, sys);
  }
  */

  return derivative_c2(&fluct_T_i, z, z * 1e-6, E, mu_e, mu_h, sys);
}

double fluct_dT_dz(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  return
    0.5 * M_PI / std::sqrt(
      0.25 * ( 1 - std::pow(sys.m_ph - sys.m_pe, 2)) * E - z - mu_e - mu_h
    ) + fluct_dT_i_dz(z, E, mu_e, mu_h, sys);
}

std::complex<double> fluct_T_i_c_flc(double a, double b) {
  return std::log(std::complex<double>((a + b) / (a - b))) / b;
}

double fluct_T_i_c_fer(double x, void * params) {
  fluct_T_i_c_s * s{static_cast<fluct_T_i_c_s*>(params)};

  if (s->E <= 1e-5 && 2 * (s->z + s->mu_e + s->mu_h) > x) {
    // In this case we compute the integral using the Cauchy principal value method.
    // Thus, there's a factor of -1 / (z + mu_e + mu_h - x) missing.
    return -2 * std::sqrt(x) / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1) * s->c_f;
  } else if (s->E <= 1e-5) {
    return 2 * std::sqrt(x) / ((s->z + s->mu_e + s->mu_h - x) * (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1)) * s->c_f;
  }

  double a_e{s->z + s->mu_e + s->mu_h - s->sys.m_pe * s->E - x};
  double b_e{2 * s->sys.m_pe * std::sqrt(s->E * x)};

  return std::sqrt(x) * (fluct_T_i_c_flc(a_e, b_e).real() / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1)) * s->c_f;
}

double fluct_T_i_c_fei(double x, void * params) {
  fluct_T_i_c_s * s{static_cast<fluct_T_i_c_s*>(params)};

  /*
  if (E < 1e-7) {
    // In this case the imaginary part can be exactly computed.
  }
  */

  double a_e{s->z + s->mu_e + s->mu_h - s->sys.m_pe * s->E - x};
  double b_e{2 * s->sys.m_pe * std::sqrt(s->E * x)};

  return std::sqrt(x) * (fluct_T_i_c_flc(a_e, b_e).imag() / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1)) * s->c_f;
}

double fluct_T_i_c_fhr(double x, void * params) {
  fluct_T_i_c_s * s{static_cast<fluct_T_i_c_s*>(params)};

  if (s->E <= 1e-5 && 2 * (s->z + s->mu_e + s->mu_h) > x) {
    // In this case we compute the integral using the Cauchy principal value method.
    // Thus, there's a factor of -1 / (z + mu_e + mu_h - x) missing.
    return - 2 * std::sqrt(x) / (std::exp((s->sys.m_pe * x - s->mu_e) / (4 * M_PI)) + 1) * s->c_f;
  } else if (s->E <= 1e-5) {
    return 2 * std::sqrt(x) / ((s->z + s->mu_e + s->mu_h - x) * (std::exp((s->sys.m_pe * x - s->mu_e) / (4 * M_PI)) + 1)) * s->c_f;
  }

  double a_e{s->z + s->mu_e + s->mu_h - s->sys.m_ph * s->E - x};
  double b_e{2 * s->sys.m_ph * std::sqrt(s->E * x)};

  return std::sqrt(x) * (fluct_T_i_c_flc(a_e, b_e).real() / (std::exp((s->sys.m_pe * x - s->mu_e) / (4 * M_PI)) + 1)) * s->c_f;
}

double fluct_T_i_c_fhi(double x, void * params) {
  fluct_T_i_c_s * s{static_cast<fluct_T_i_c_s*>(params)};

  /*
  if (E < 1e-7) {
    // In this case the imaginary part can be exactly computed.
  }
  */

  double a_e{s->z + s->mu_e + s->mu_h - s->sys.m_pe * s->E - x};
  double b_e{2 * s->sys.m_pe * std::sqrt(s->E * x)};

  return std::sqrt(x) * (fluct_T_i_c_flc(a_e, b_e).imag() / (std::exp((s->sys.m_ph * x - s->mu_h) / (4 * M_PI)) + 1)) * s->c_f;
}

uint32_t fluct_T_i_c_fb(double * pts, double z, double E, double m_pi, double mu_t) {
  /*
   * Computes the bounds for the branch frequency integral.
   */
  double sqrt_arg{m_pi * (m_pi - 1.0) * E + z + mu_t};

  if (sqrt_arg > global_eps) {
    pts[1] = m_pi * (2.0 * m_pi - 1.0) * E + z + mu_t - 2 * m_pi * std::sqrt(E * sqrt_arg);
    pts[2] = m_pi * (2.0 * m_pi - 1.0) * E + z + mu_t + 2 * m_pi * std::sqrt(E * sqrt_arg);
    pts[3] = 2 * pts[2];

    return 2;
  } else if (sqrt_arg >= 0 && sqrt_arg <= global_eps) {
    pts[1] = m_pi * (2.0 * m_pi - 1.0) * E + z + mu_t;
    pts[2] = 2 * pts[1];

    return 1;
  }

  return 0;
}

std::complex<double> fluct_T_c(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{6};
  constexpr uint32_t n_pts{4};

  double mu_t{mu_e + mu_h};
  fluct_T_i_c_s s{ z, E, mu_e, mu_h, sys };

  double pts_e[n_pts] = {0}, pts_h[n_pts] = {0};
  size_t neval[1] = {0};
  double result[n_int] = {0}, error[n_int] = {0};

  gsl_function integrand_er;
  integrand_er.function = &fluct_T_i_c_fer;
  integrand_er.params = &s;

  gsl_function integrand_hr;
  integrand_hr.function = &fluct_T_i_c_fhr;
  integrand_hr.params = &s;

  /*
    If E < global_eps we use different methods to compute the
    integral. The real part can be approximated by Cauchy principal
    values integration, and the imaginary part is exact.

    Also, z + mu_t has to be positive, otherwise the integral
    is just given by fluct_i(...).
  */

  pts_e[1] = z + mu_t;
  if (E <= 1e-5 && pts_e[1] > 0) {
    constexpr uint32_t local_ws_size{(1<<4)};

    // Yes, it's ok to use pts_e for both integrals.
    pts_e[2] = 2 * pts_e[1];

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qawc(&integrand_er, pts_e[0], pts_e[2], pts_e[1], 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_qagiu(&integrand_er, pts_e[2], 0, global_eps, local_ws_size, ws, result + 1, error + 1);

    gsl_integration_qawc(&integrand_hr, pts_e[0], pts_e[2], pts_e[1], 0, global_eps, local_ws_size, ws, result + 2, error + 2);
    gsl_integration_qagiu(&integrand_hr, pts_e[2], 0, global_eps, local_ws_size, ws, result + 3, error + 3);

    gsl_integration_workspace_free(ws);

    /*
    for (uint32_t i = 0; i < n_int; i++) {
      printf("results: %.10f, %.3e\n", result[i],  error[i]);
    }
    */

    return std::complex<double>(
      result[0] + result[1] + result[2] + result[3],
      -M_PI_2 * M_SQRT1_2 * std::sqrt(pts_e[1]) * (1 / (std::exp((sys.m_ph * pts_e[1] - mu_h) / (4 * M_PI)) + 1) + 1 / (std::exp((sys.m_pe * pts_e[1] - mu_e)/(4 * M_PI)) + 1))
    );
  }

  /*
    First we compute the singular points of the integrand.

    case 0: If there are none, then fluct_i_c == fluct_i.
    case 1: If there is just one, the imaginary part is exactly zero.
    case 2: If there are two, then there is a contribution to both
    real and imaginary parts.
  */

  uint32_t int_state_e{fluct_T_i_c_fb(pts_e, z, E, sys.m_pe, mu_t)};
  uint32_t int_state_h{fluct_T_i_c_fb(pts_h, z, E, sys.m_ph, mu_t)};

  gsl_function integrand_ei;
  integrand_ei.function = &fluct_T_i_c_fei;
  integrand_ei.params = &s;

  gsl_function integrand_hi;
  integrand_hi.function = &fluct_T_i_c_fhi;
  integrand_hi.params = &s;

  constexpr uint32_t local_ws_size{(1<<4)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  switch (int_state_e) {
    case 0:
      gsl_integration_qagiu(&integrand_er, pts_e[0], 0, global_eps, local_ws_size, ws, result, error);
      break;

    case 1:
      gsl_integration_qagp(&integrand_er, pts_e, 3, 0, global_eps, local_ws_size, ws, result, error);
      gsl_integration_qagiu(&integrand_er, pts_e[2], 0, global_eps, local_ws_size, ws, result + 1, error + 1);
      break;

    case 2:
      gsl_integration_qagp(&integrand_er, pts_e, n_pts, 0, global_eps, local_ws_size, ws, result, error);
      gsl_integration_qagiu(&integrand_er, pts_e[3], 0, global_eps, local_ws_size, ws, result + 1, error + 1);
      gsl_integration_qng(&integrand_ei, pts_e[1], pts_e[2], 0, global_eps, result + 2, error + 2, neval);
      break;
  }

  switch (int_state_h) {
    case 0:
      gsl_integration_qagiu(&integrand_hr, pts_h[0], 0, global_eps, local_ws_size, ws, result + 3, error + 3);
      break;

    case 1:
      gsl_integration_qagp(&integrand_hr, pts_h, 3, 0, global_eps, local_ws_size, ws, result + 3, error + 3);
      gsl_integration_qagiu(&integrand_hr, pts_h[2], 0, global_eps, local_ws_size, ws, result + 4, error + 4);
      break;

    case 2:
      gsl_integration_qagp(&integrand_hr, pts_h, n_pts, 0, global_eps, local_ws_size, ws, result + 3, error + 3);
      gsl_integration_qagiu(&integrand_hr, pts_h[3], 0, global_eps, local_ws_size, ws, result + 4, error + 4);
      gsl_integration_qng(&integrand_hi, pts_h[1], pts_h[2], 0, global_eps, result + 5, error + 5, neval);
      break;
  }

  gsl_integration_workspace_free(ws);

  return std::complex<double>(result[0]+result[1]+result[3]+result[4], result[2]+result[5]);
}

double fluct_pp_f(double z, void * params) {
  struct fluct_pp_s * s{static_cast<struct fluct_pp_s*>(params)};

  return fluct_T(z, s->E, s->a, s->mu_e, s->mu_h, s->sys);
}

double fluct_pp_df(double z, void * params) {
  struct fluct_pp_s * s{static_cast<struct fluct_pp_s*>(params)};

  return fluct_dT_dz(z, s->E, s->mu_e, s->mu_h, s->sys);
}

void fluct_pp_fdf(double z, void * params, double * f, double * df) {
  f[0] = fluct_pp_f(z, params);
  df[0] = fluct_pp_df(z, params);
}

double fluct_pp_s(double E, double a, double mu_e, double mu_h, const system_data & sys) {
  /*
    Computes the value z0 that satisfies the following:

    pi / (2 * a) + pi * sqrt((1-(mr_p / mr_m)^2)*E/4 - z0 - mu_e - mu_h) + fluct_es_f(z0, ...) == 0
  */
  struct fluct_pp_s params{E, a, mu_e, mu_h, sys};

  double z1{sys.get_z1(E, mu_e + mu_h)};

  double T_at_z1{fluct_T_z1(E, a, mu_e, mu_h, sys)};

  if (T_at_z1 < 0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if (T_at_z1 == 0) {
    return z1;
  }

  double z{z1 - 3 / 16 * std::pow(a - fluct_ac(E, mu_e, mu_h, sys), 2)}, z0;

  gsl_function_fdf funct;
  funct.f = &fluct_pp_f;
  funct.df = &fluct_pp_df;
  funct.fdf = &fluct_pp_fdf;
  funct.params = &params;

  const gsl_root_fdfsolver_type * T = gsl_root_fdfsolver_steffenson;
  gsl_root_fdfsolver * s = gsl_root_fdfsolver_alloc(T);

  gsl_root_fdfsolver_set(s, &funct, z);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fdfsolver_iterate(s);
    z0 = z;

    z = gsl_root_fdfsolver_root(s);
    status = gsl_root_test_delta(z, z0, 0, global_eps);
  }

  gsl_root_fdfsolver_free(s);
  return z;
}

double fluct_pp_b(double E, double a, double mu_e, double mu_h, const system_data & sys) {
  /*
    Computes the value z0 that satisfies the following:

    pi / (2 * a) + pi * sqrt((1-(mr_p / mr_m)^2)*E/4 - z0 - mu_e - mu_h) + fluct_es_f(z0, ...) == 0

    Brent version.
  */
  struct fluct_pp_s params{E, a, mu_e, mu_h, sys};

  double z1{sys.get_z1(E, mu_e + mu_h)};

  double T_at_z1{fluct_T_z1(E, a, mu_e, mu_h, sys)};

  if (T_at_z1 < 0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if (T_at_z1 == 0) {
    return z1;
  }

  double z{z1 - 0.25 * std::pow(a - fluct_ac(E, mu_e, mu_h, sys), 2)};
  double z_min{z}, z_max{0.5 * (z1 + z)};

  gsl_function funct;
  funct.function = &fluct_pp_f;
  funct.params = &params;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pp(double E, double a, double mu_e, double mu_h, const system_data & sys) {
  return fluct_pp_b(E, a, mu_e, mu_h, sys);
}

double fluct_pp0_E_f(double E, void * params) {
  fluct_pp0_E_s * s{static_cast<fluct_pp0_E_s*>(params)};
  return fluct_pp(E, s->a, s->mu_e, s->mu_h, s->sys);
}

double fluct_pp0_E(double a, double mu_e, double mu_h, const system_data & sys) {
  /*
    Computes the energy E at which z0(E, a) == 0.
    Brent version.
  */
  fluct_pp0_E_s params{a, mu_e, mu_h, sys};

  double ac{fluct_ac(0, mu_e, mu_h, sys)};
  double z, z_min{0};
  double z_max{(4 * std::abs(mu_e + mu_h) + std::pow(a - ac, 2)) / (1 - std::pow(sys.m_pe - sys.m_ph, 2)) + 1};

  double f_min{fluct_pp0_E_f(z_min, &params)};

  if (f_min > 0 || std::isnan(f_min)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  gsl_function funct;
  funct.function = &fluct_pp0_E_f;
  funct.params = &params;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pp0_a_f(double a, void * params) {
  fluct_pp0_a_s * s{static_cast<fluct_pp0_a_s*>(params)};
  return fluct_pp(a, 0, s->mu_e, s->mu_h, s->sys);
}

double fluct_pp0_a(double mu_e, double mu_h, const system_data & sys) {
  /*
    Computes the scattering length a
    at which z0(0, a) == 0. Brent version.
  */
  fluct_pp0_a_s params{mu_e, mu_h, sys};

  if (mu_e + mu_h > 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double ac{fluct_ac(0, mu_e, mu_h, sys)};
  double z, z_max{2 * std::sqrt(-(mu_e + mu_h))};
  double z_min{z_max + ac};

  gsl_function funct;
  funct.function = &fluct_pp0_a_f;
  funct.params = &params;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pp0_mu_f(double mu_e, void * params) {
  fluct_pp0_mu_s * s{static_cast<fluct_pp0_mu_s*>(params)};

  double mu_h{ideal_mu_h(mu_e, s->sys)};
  return fluct_pp0_a(mu_e, mu_h, s->sys) - s->a;
}

double fluct_pp0_mu(double a, double n, const system_data & sys) {
  /*
    Computes the value of mu_e such that ac_max == a.
    Here we compute mu_h(mu_e) assuming n_id,e == n_id,h.
  */
  fluct_pp0_mu_s params{a, sys};
  double z, z_min{-0.25 *a*a * (a > 0) + 4 * M_PI * invPolylogExp(1.5, 0.25 * std::pow(sys.m_sigma, -1.5) * n)};
  double z_max{ideal_mu_v(0, sys)};

  if (a < fluct_pp0_a(z_max, ideal_mu_h(z_max, sys), sys)) {
    // In this case there is no root.
    return std::numeric_limits<double>::quiet_NaN();
  }

  gsl_function funct;
  funct.function = &fluct_pp0_mu_f;
  funct.params = &params;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_pr(double E, double a, double mu_e, double mu_h, const system_data & sys) {
  double z0{fluct_pp(E, a, mu_e, mu_h, sys)};

  if (std::isnan(z0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double z1{sys.get_z1(E, mu_e + mu_h)};
  double sqrt_val{2 * std::sqrt(z1 - z0) * M_1_PI};

  return 1 / ( (std::exp(z0 / (4 * M_PI)) - 1) * (1 + sqrt_val * fluct_dT_i_dz(z0, E, mu_e, mu_h, sys)));
}

double fluct_ac(double E, double mu_e, double mu_h, const system_data & sys) {
  return - 2 * M_1_PI * fluct_T_z1_i(E, mu_e, mu_h, sys);
}

double fluct_Ec_f(double E, void * params) {
  fluct_Ec_s * s{static_cast<fluct_Ec_s*>(params)};
  return fluct_ac(E, s->mu_e, s->mu_h, s->sys) - s->a;
}

double fluct_Ec(double a, double mu_e, double mu_h, const system_data & sys) {
  /*
    Given some scattering length a, E_c(a) is the maximum kinetic energy
    that will contain excitons. It satisfies E_c(a_c) = 0.
  */
  fluct_Ec_s params{a, mu_e, mu_h, sys};

  double ac{fluct_ac(0, mu_e, mu_h, sys)};

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

  for (uint32_t i = 0; i < max_iter; z_max *= 8) {
    // As E increases, a_c(E) -> 0, so at some point a_c(E) < a.
    if (fluct_Ec_f(z_max, &params) > 0) {
      break;
    }
    z_min = z_max;
  }

  gsl_function funct;
  funct.function = &fluct_Ec_f;
  funct.params = &params;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(s);
  return z;
}

double fluct_n_ex_f(double E, void * params) {
  fluct_n_ex_s * s{static_cast<fluct_n_ex_s*>(params)};

  double pr{fluct_pr(E, s->a, s->mu_e, s->mu_h, s->sys)};

  if (std::isnan(pr)) {
    return 0;
  }

  return s->c_f * std::sqrt(E) * pr;
}

double fluct_n_ex(double a, double mu_e, double mu_h, const system_data & sys) {
  /*
    Assuming a < ac_max, therefore no singularities in the integrand.
  */
  double Emax{fluct_Ec(a, mu_e, mu_h, sys)};

  if (std::isnan(Emax)) {
    // a < a_c
    return 0;
  }

  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};
  fluct_n_ex_s params{a, mu_e, mu_h, sys};

  gsl_function integrand;
  integrand.function = &fluct_n_ex_f;
  integrand.params = &params;

  if (std::isinf(Emax)) {
    constexpr uint32_t local_ws_size{1<<3};
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_workspace_free(ws);
  } else {
    constexpr uint32_t local_ws_size{1<<3};
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    gsl_integration_qag(&integrand, 0, Emax, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result, error);
    gsl_integration_workspace_free(ws);
  }

  return result[0];
}

double fluct_n_ex_c(double a, double mu_e, double mu_h, const system_data & sys) {
  double Emax{fluct_Ec(a, mu_e, mu_h, sys)};

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

  constexpr uint32_t n_int{2};
  double result[n_int] = {0}, error[n_int] = {0};
  fluct_n_ex_s params{a, mu_e, mu_h, sys};

  gsl_function integrand;
  integrand.function = &fluct_n_ex_f;
  integrand.params = &params;

  double ac_max{fluct_pp0_a(mu_e, mu_h, sys)};

  if (a < ac_max) {
    if (std::isinf(Emax)) {
      constexpr uint32_t local_ws_size{1<<3};
      gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
      gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
      gsl_integration_workspace_free(ws);
    } else {
      constexpr uint32_t local_ws_size{1<<3};
      gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
      gsl_integration_qag(&integrand, 0, Emax, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result, error);
      gsl_integration_workspace_free(ws);
    }
  } else {
    double Epole{fluct_pp0_E(a, mu_e, mu_h, sys)};
    constexpr uint32_t local_ws_size{1<<4};
    double pts[] = {0, Epole, 2 * Epole};

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    gsl_integration_qagp(&integrand, pts, 3, 0, global_eps, local_ws_size, ws, result, error);

    gsl_integration_qagiu(&integrand, pts[2], 0, global_eps, local_ws_size, ws, result + 1, error + 1);
    gsl_integration_workspace_free(ws);
  }

  return sum_result<n_int>(result);
}

double fluct_bfi_f(double y, void * params) {
  fluct_bfi_s * s{static_cast<fluct_bfi_s*>(params)};

  double y_sq{y*y};
  double new_var{y_sq + s->z1};

  std::complex<double> I2{fluct_T_c(new_var, s->E, s->mu_e, s->mu_h, s->sys) * M_1_PI};
  return (s->a + 2 * I2.real()) / (std::norm(std::complex<double>(0.5 * s->a, y) + I2) * (std::exp(new_var / (4 * M_PI)) - 1));
}

double fluct_bfi(double E, double a, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{2};
  constexpr double x_max{1};
  double result[n_int] = {0}, error[n_int] = {0};

  double z1{sys.get_z1(E, mu_e + mu_h)};
  fluct_bfi_s params{E, a, mu_e, mu_h, z1, sys};

  gsl_function integrand;
  integrand.function = &fluct_bfi_f;
  integrand.params = &params;

  constexpr uint32_t local_ws_size{1<<3};
  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  gsl_integration_qag(&integrand, 0, x_max, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS21, ws, result, error);
  gsl_integration_qagiu(&integrand, x_max, 0, global_eps, local_ws_size, ws, result + 1, error + 1);

  gsl_integration_workspace_free(ws);

  return sum_result<n_int>(result);
}

double fluct_n_sc_f(double E, void * params) {
  fluct_n_sc_s * s = static_cast<fluct_n_sc_s*>(params);
  return s->c_f * std::sqrt(E) * fluct_bfi(E, s->a, s->mu_e, s->mu_h, s->sys);
}

double fluct_n_sc(double a, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{2};

  double Emax{fluct_Ec(a, mu_e, mu_h, sys)};

  if (std::isnan(Emax)) {
    // a < a_c
    Emax = std::numeric_limits<double>::infinity();
  }

  double result[n_int] = {0}, error[n_int] = {0};

  fluct_n_sc_s params({a, mu_e, mu_h, sys});

  gsl_function integrand;
  integrand.function = &fluct_n_sc_f;
  integrand.params = &params;

  if (std::isinf(Emax)) {
    constexpr uint32_t local_ws_size{1<<3};
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_workspace_free(ws);
  } else {
    constexpr uint32_t local_ws_size{1<<3};
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
    gsl_integration_qag(&integrand, 0, Emax, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result, error);
    gsl_integration_qagiu(&integrand, Emax, 0, global_eps, local_ws_size, ws, result + 1, error + 1);
    gsl_integration_workspace_free(ws);
  }

  return sum_result<n_int>(result);
}

template<bool N> double fluct_dn_f(double x1, double x2, const system_data & sys) {
  double mu_e, a;

  if constexpr(N) {
    mu_e = x1;
    a = x2;
  } else {
    a = x1;
    mu_e = x2;
  }

  double mu_h{ideal_mu_h(mu_e, sys)};
  return fluct_n_ex(a, mu_e, mu_h, sys) + fluct_n_sc(a, mu_e, mu_h, sys);
}

double fluct_dn_dmu(double mu_e, double a, const system_data & sys) {
  return derivative_c2(&fluct_dn_f<true>, mu_e, mu_e * 1e-6, a, sys);
}

double fluct_dn_da(double mu_e, double a, const system_data & sys) {
  return derivative_c2(&fluct_dn_f<false>, a, a * 1e-6, mu_e, sys);
}

int fluct_mu_f(const gsl_vector * x, void * params, gsl_vector * f) {
  constexpr uint32_t n_eq{2};
  // defined in analytic_utils.h
  fluct_mu_s * s{static_cast<fluct_mu_s*>(params)};

  double mu_e{gsl_vector_get(x, 0)};
  double a{gsl_vector_get(x, 1)};

  double mu_h{ideal_mu_h(mu_e, s->sys)};

  double n_id{2*ideal_n(mu_e, s->sys.m_pe)};

  double ls{ideal_ls(n_id, s->sys)};
  double new_a{analytic_a_ls(ls, s->sys)};

  double yv[n_eq] = {0};
  yv[0] = - s->n + n_id + fluct_n_ex(new_a, mu_e, mu_h, s->sys) + fluct_n_sc(new_a, mu_e, mu_h, s->sys);
  yv[1] = - a + new_a;
  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }

  return GSL_SUCCESS;
}

std::vector<double> fluct_mu_f(double a, double mu_e, double n, const system_data & sys) {
  gsl_vector * x = gsl_vector_alloc(2);
  gsl_vector_set(x, 0, mu_e);
  gsl_vector_set(x, 1, a);

  fluct_mu_s params{n, sys};

  fluct_mu_f(x, &params, x);

  std::vector<double> r(2);
  for (uint32_t i = 0; i < 2; i++) { r[i] = gsl_vector_get(x, i); }

  gsl_vector_free(x);
  return r;
}

template<uint32_t N, uint32_t M> double fluct_mu_df_nm(double mu_e, double a, const fluct_mu_s * s) {
  if constexpr(N == 0) {
    if constexpr(M == 0) {
      /*
       * dn_eq/dmu
       */
      return 2 * ideal_dn_dmu(mu_e, s->sys.m_pe) + fluct_dn_dmu(mu_e, a, s->sys);

    } else if constexpr(M == 1) {
      /*
       * dn_eq/da
       */
      return fluct_dn_da(mu_e, a, s->sys);

    }
  } else if constexpr(N == 1) {
    if constexpr(M == 0) {
      /*
       * da_eq/dmu
       */
      double n_id{2*ideal_n(mu_e, s->sys.m_pe)};
      double ls{ideal_ls(n_id, s->sys)};
      return derivative_c2(&analytic_a_ls, ls, ls * 1e-6, s->sys);

    } else if constexpr(M == 1) {
      /*
       * da_eq/da
       */
      return 0;

    }
  }
}

int fluct_mu_df(const gsl_vector * x, void * params, gsl_matrix * J) {
  // defined in fluct_utils.h
  fluct_mu_s * s{static_cast<fluct_mu_s*>(params)};

  double mu_e{gsl_vector_get(x, 0)};
  double a{gsl_vector_get(x, 1)};

  gsl_matrix_set(J, 0, 0, fluct_mu_df_nm<0, 0>(mu_e, a, s));
  gsl_matrix_set(J, 0, 1, fluct_mu_df_nm<0, 1>(mu_e, a, s));
  gsl_matrix_set(J, 1, 0, fluct_mu_df_nm<1, 0>(mu_e, a, s));
  gsl_matrix_set(J, 1, 1, fluct_mu_df_nm<1, 1>(mu_e, a, s));

  return GSL_SUCCESS;
}

int fluct_mu_fdf(const gsl_vector * x, void * params, gsl_vector * f, gsl_matrix * J) {
  fluct_mu_f(x, params, f);
  fluct_mu_df(x, params, J);

  return GSL_SUCCESS;
}

std::vector<double> fluct_mu(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  fluct_mu_s params_s{n, sys};

  gsl_multiroot_function_fdf f = {
    &fluct_mu_f,
    &fluct_mu_df,
    &fluct_mu_fdf,
    n_eq,
    &params_s
  };

  double init_a{-1};
  double x_init[n_eq] = {
    analytic_mu_init_mu(n, init_a, sys),
    init_a
  };

  printf("first: %.3f, %.10f\n", n, x_init[0]);

  gsl_vector * x = gsl_vector_alloc(n_eq);

  for(size_t i = 0; i < n_eq; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fdfsolver_type * T = gsl_multiroot_fdfsolver_hybridsj;
  gsl_multiroot_fdfsolver * s = gsl_multiroot_fdfsolver_alloc(T, n_eq);
  gsl_multiroot_fdfsolver_set(s, &f, x);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    double mu_e{gsl_vector_get(s->x, 0)}, a{gsl_vector_get(s->x, 1)};
    printf("iter %d: %.3f, %.10f, %.10f, %.6f\n", iter, n, mu_e, a, fluct_pp0_a(mu_e, ideal_mu_h(mu_e, sys), sys));
    status = gsl_multiroot_fdfsolver_iterate(s);
    if (status) { break; }
    //status = gsl_multiroot_test_residual(s->f, global_eps);
    status = gsl_multiroot_test_delta(s->dx, s->x, 0, global_eps);
  }

  std::vector<double> r(n_eq);

  for(size_t i = 0; i < n_eq; i++) { r[i] = gsl_vector_get(s->x, i); }

  gsl_multiroot_fdfsolver_free(s);
  gsl_vector_free(x);

  return r;
}

std::vector<double> fluct_mu_steps(double n, std::vector<double> x_init, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  fluct_mu_s params_s{n, sys};

  gsl_multiroot_function_fdf f = {
    &fluct_mu_f,
    &fluct_mu_df,
    &fluct_mu_fdf,
    n_eq,
    &params_s
  };

  gsl_vector * x = gsl_vector_alloc(n_eq);

  printf("first: %.3f, %.10f\n", n, x_init[0]);
  for(size_t i = 0; i < n_eq; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fdfsolver_type * T = gsl_multiroot_fdfsolver_hybridsj;
  gsl_multiroot_fdfsolver * s = gsl_multiroot_fdfsolver_alloc(T, n_eq);
  gsl_multiroot_fdfsolver_set(s, &f, x);

  std::vector<double> r;
  for(size_t i = 0; i < n_eq; i++) {
    r.push_back(gsl_vector_get(s->x, i));
    r.push_back(gsl_vector_get(s->f, i));
  }

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    printf("iter %d: %.3f, %.10f, %.10f\n", iter, n, gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));
    status = gsl_multiroot_fdfsolver_iterate(s);
    if (status) { break; }
    //status = gsl_multiroot_test_residual(s->f, global_eps);
    status = gsl_multiroot_test_delta(s->dx, s->x, 0, global_eps);

    for(size_t i = 0; i < n_eq; i++) {
      r.push_back(gsl_vector_get(s->x, i));
      r.push_back(gsl_vector_get(s->f, i));
    }
  }

  for(size_t i = 0; i < n_eq; i++) {
    r.push_back(gsl_vector_get(s->x, i));
    r.push_back(gsl_vector_get(s->f, i));
  }

  gsl_multiroot_fdfsolver_free(s);
  gsl_vector_free(x);

  return r;
}

