#include "fluctuations_2d.h"
#include "fluctuations_2d_utils.h"

double fluct_2d_I2_f1(double z, double a, double b) {
  /*
   * This is the solution of the angular integral
   * that contributes to the fluctuations part of
   * the T-matrix.
   */

  return sign(z - a + b) / std::sqrt(std::abs(std::pow(a - z, 2) - b*b));
}

double fluct_2d_I2_f2e(double x, double z, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Here we introduce the Fermi-Dirac distribution
   * in the momentum integral.
   *
   * Electron contribution.
   */

  return
    fluct_2d_I2_f1(z, - mu_e - mu_h + sys.m_ph * E + x, 2 * sys.m_ph * std::sqrt(E * x))
    / (std::exp((sys.m_pe * x - mu_e) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_f2h(double x, double z, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Here we introduce the Fermi-Dirac distribution
   * in the momentum integral.
   *
   * Hole contribution.
   */

  return
    fluct_2d_I2_f1(z, - mu_e - mu_h + sys.m_pe * E + x, 2 * sys.m_pe * std::sqrt(E * x))
    / (std::exp((sys.m_ph * x - mu_h) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_f(double x, void * params) {
  /*
   * Integrand of I2.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_s * s{static_cast<fluct_2d_I2_s*>(params)};

  return fluct_2d_I2_f2e(x, s->z, s->E, s->mu_e, s->mu_h, s->sys)
    + fluct_2d_I2_f2h(x, s->z, s->E, s->mu_e, s->mu_h, s->sys);
}

double fluct_2d_I2(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes I2.
   * If z < z1 then it is real, otherwise it will have an imaginary part.
   */
  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};

  fluct_2d_I2_s s{z, E, mu_e, mu_h, sys};

  gsl_function integrands[n_int];

  integrands[0].function = &fluct_2d_I2_f;
  integrands[0].params = &s;

  constexpr uint32_t local_ws_size{(1<<8)};

  //printf("bounds: (%f, %f), (%f, %f)\n", bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]);

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(integrands, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return
    // Re(I2)_0^inf
    0.25 * M_1_PI * result[0]
  ;
}

double fluct_2d_I2_dz_f(double y, void * params) {
  /*
   * Integrand of d(I2)/dz.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_dz_s * s{static_cast<fluct_2d_I2_dz_s*>(params)};

  double ye_sqrt{(std::sqrt(y) - 1)};
  double yh_sqrt{ye_sqrt * s->yh_min};
  ye_sqrt *= s->ye_min;

  return
      s->ye_min2 / (std::exp((s->sys.m_pe * ye_sqrt - s->mu_e) * 0.25 * M_1_PI) + 1)
    * std::pow(y * s->ye_min2 - 4 * s->sys.m_ph*s->sys.m_ph * s->E * ye_sqrt, -1.5)
    + s->yh_min2 / (std::exp((s->sys.m_ph * yh_sqrt - s->mu_h) * 0.25 * M_1_PI) + 1)
    * std::pow(y * s->yh_min2 - 4 * s->sys.m_pe*s->sys.m_pe * s->E * yh_sqrt, -1.5);
}

double fluct_2d_I2_dz_f2(double y, void * params) {
  /*
   * Integrand of d(I2)/dz.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_dz_s * s{static_cast<fluct_2d_I2_dz_s*>(params)};

  return
      1 / (std::exp((s->sys.m_pe * y*y - s->mu_e) * 0.25 * M_1_PI) + 1)
    * std::pow(y*y + s->ye_min, -1.5)
    + 1 / (std::exp((s->sys.m_ph * y*y - s->mu_h) * 0.25 * M_1_PI) + 1)
    * std::pow(y*y + s->yh_min, -1.5);
}

double fluct_2d_I2_dz(double z0, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes d(I2)/dz.
   * If z < z1 then it is real, otherwise it will have an imaginary part.
   */
  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};

  fluct_2d_I2_dz_s s{
    - mu_e - mu_h + sys.m_ph * E - z0,
    - mu_e - mu_h + sys.m_pe * E - z0,
    0,
    0,
    z0, E, mu_e, mu_h, sys
  };

  s.ye_min2 = s.ye_min * s.ye_min;
  s.yh_min2 = s.yh_min * s.yh_min;

  gsl_function integrands[2];

  integrands[0].function = &fluct_2d_I2_dz_f;
  integrands[0].params = &s;

  constexpr uint32_t local_ws_size{(1<<8)};

  //printf("bounds: (%f, %f), (%f, %f)\n", bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]);

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  if (s.ye_min > global_eps) {
    gsl_integration_qagiu(integrands, 1, 0, global_eps, local_ws_size, ws, result, error);

    result[0] *= - 0.125 * M_1_PI;
  } else {
    result[0] = - std::numeric_limits<double>::infinity();
  }

  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return
    // Re(I2)_0^inf
    result[0]
  ;
}

double fluct_2d_I2_p_fp(double y, void * params) {
  /*
   * Integrand of I2.
   * x > x_+.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_p_s * s{static_cast<fluct_2d_I2_p_s*>(params)};

  return 1 / (std::exp((s->sys.m_pe * (s->x_0e + s->dx_0e * std::cosh(2 * y)) - s->mu_e) * 0.25 * M_1_PI) + 1)
    + 1 / (std::exp((s->sys.m_ph * (s->x_0h + s->dx_0h * std::cosh(2 * y)) - s->mu_h) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_p_fm(double y, void * params) {
  /*
   * Integrand of I2.
   * 0 < x < x_-.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_p_s * s{static_cast<fluct_2d_I2_p_s*>(params)};

  return s->sign_e * s->ye_max / (std::exp((s->sys.m_pe * (s->x_0e - s->dx_0e * std::cosh(2 * y * s->ye_max)) - s->mu_e) * 0.25 * M_1_PI) + 1)
    + s->sign_h * s->yh_max / (std::exp((s->sys.m_ph * (s->x_0h - s->dx_0h * std::cosh(2 * y * s->yh_max)) - s->mu_h) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_p_fi(double th, void * params) {
  /*
   * Integrand of I2.
   * x_- < x < x_+.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_p_s * s{static_cast<fluct_2d_I2_p_s*>(params)};

  return 1 / (std::exp((s->sys.m_pe * (s->x_0e - s->dx_0e * std::cos(th)) - s->mu_e) * 0.25 * M_1_PI) + 1)
    + 1 / (std::exp((s->sys.m_ph * (s->x_0h - s->dx_0h * std::cos(th)) - s->mu_h) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_p_fcr(double x, void * params) {
  /*
   * Real part of the integral in the limit E == 0.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_p_s * s{static_cast<fluct_2d_I2_p_s*>(params)};

  return 1 / (std::exp((s->sys.m_pe * x - s->mu_e) * 0.25 * M_1_PI) + 1)
    + 1 / (std::exp((s->sys.m_ph * x - s->mu_h) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_p_fr(double x, void * params) {
  /*
   * Real part of the integral in the limit E == 0.
   * Without Cauchy 1 / (x-x0).
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_p_s * s{static_cast<fluct_2d_I2_p_s*>(params)};

  return (1 / (std::exp((s->sys.m_pe * x - s->mu_e) * 0.25 * M_1_PI) + 1)
    + 1 / (std::exp((s->sys.m_ph * x - s->mu_h) * 0.25 * M_1_PI) + 1))
    / ( x - s->x_0e );
}

double fluct_2d_I2_p_i(fluct_2d_I2_p_s * s) {
  /*
   * Real part of the integral in the limit E == 0.
   * With Cauchy 1 / (x-x0).
   */

  return 1 / (std::exp((s->sys.m_pe * s->x_0e - s->mu_e) * 0.25 * M_1_PI) + 1)
    + 1 / (std::exp((s->sys.m_ph * s->x_0h - s->mu_h) * 0.25 * M_1_PI) + 1);
}

std::complex<double> fluct_2d_I2_p(double z, double E, double chi_ex, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes I2(E - mu_t - chi_ex * z, sys.m_sigma * E).
   * If z < z1 then it is real, otherwise it will have an imaginary part.
   */

  constexpr uint32_t n_int{3};
  double result[n_int] = {0}, error[n_int] = {0};
  //size_t neval[n_int] = {0};
  //double dt[n_int] = {0};

  fluct_2d_I2_p_s s{
    sys.m_ph / sys.m_pe * E - chi_ex * z,
    2 * std::sqrt(- chi_ex * z * sys.m_ph / sys.m_pe * E),
    0,
    0,
    0,

    sys.m_pe / sys.m_ph * E - chi_ex * z,
    2 * std::sqrt(- chi_ex * z * sys.m_pe / sys.m_ph * E),
    0,
    0,
    0,

    - sign(sys.m_ph / sys.m_pe * E + chi_ex * z),
    - sign(sys.m_pe / sys.m_ph * E + chi_ex * z),

    E,
    mu_e,
    mu_h,
    sys
  };

  s.x_me = s.x_0e - s.dx_0e;
  s.x_pe = s.x_0e + s.dx_0e;

  s.x_mh = s.x_0h - s.dx_0h;
  s.x_ph = s.x_0h + s.dx_0h;

  s.ye_max = std::asinh(std::sqrt(0.5 * s.x_me / s.dx_0e));
  s.yh_max = std::asinh(std::sqrt(0.5 * s.x_mh / s.dx_0h));

  if (z == 0) {
    return {
      -std::numeric_limits<double>::infinity(),
      0.125 * M_1_PI * fluct_2d_I2_p_i(&s)
    };
  }

  gsl_function integrands[5];

  integrands[0].function = &fluct_2d_I2_p_fp;
  integrands[0].params = &s;

  integrands[1].function = &fluct_2d_I2_p_fm;
  integrands[1].params = &s;

  integrands[2].function = &fluct_2d_I2_p_fi;
  integrands[2].params = &s;

  integrands[3].function = &fluct_2d_I2_p_fcr;
  integrands[3].params = &s;

  integrands[4].function = &fluct_2d_I2_p_fr;
  integrands[4].params = &s;

  //printf("x_me: %f, x_mh: %f, x_pe: %f, x_ph: %f\n", s.x_me, s.x_mh, s.x_pe, s.x_ph);

  constexpr uint32_t local_ws_size{(1<<5)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  if (s.dx_0e > global_eps) {
    // x > x_+.
    //auto t0{std::chrono::high_resolution_clock::now()};
    gsl_integration_qagiu(integrands, 0, 0, global_eps, local_ws_size, ws, result, error);
    //dt[0] = -(t0 - std::chrono::high_resolution_clock::now()).count();

    // 0 < x < x_-.
    //t0 = std::chrono::high_resolution_clock::now();
    gsl_integration_qag(integrands + 1, 0, 1, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS61, ws, result + 1, error + 1);
    //dt[1] = -(t0 - std::chrono::high_resolution_clock::now()).count();

    // x_- < x < x_+.
    //t0 = std::chrono::high_resolution_clock::now();
    gsl_integration_qag(integrands + 2, 0, M_PI, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS61, ws, result + 2, error + 2);
    //dt[2] = -(t0 - std::chrono::high_resolution_clock::now()).count();
  } else {
    // In this case, s.x_0e == s.x_0h.
    gsl_integration_qagiu(integrands + 4, 2 * s.x_0e, 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_qawc(integrands + 3, 0, 2 * s.x_0e, s.x_0e, 0, global_eps, local_ws_size, ws, result + 1, error + 1);
    result[2] = fluct_2d_I2_p_i(&s);
  }

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e), %.2f μs\n", i, result[i], error[i], 1e-3 * dt[i]);
  }
  printf("\n");
  */

  gsl_integration_workspace_free(ws);

  return {
    0.5 * M_1_PI * (-result[0] + result[1]),
    0.125 * M_1_PI * result[2]
  };
}

double fluct_2d_pp_f(double z, void * params) {
  struct fluct_2d_pp_s * s{static_cast<fluct_2d_pp_s*>(params)};

  return z - s->z1 - s->chi_ex * std::exp(2 * fluct_2d_I2(z, s->E, s->mu_e, s->mu_h, s->sys));
}

double fluct_2d_pp(double E, double chi_ex, double mu_e, double mu_h, const system_data & sys) {
  /*
   * The position of the pole is defined by
   *
   * z0 == z1 - chi_ex * exp(2 * I2(z0))
   *
   * Since I2 < 0, it is bounded: z1 + chi_ex < z0 < zi.
   */

  double z1{sys.get_z1(E, mu_e + mu_h)};

  struct fluct_2d_pp_s params{z1, E, chi_ex, mu_e, mu_h, sys};

  double z{0};
  double z_min{z1 + chi_ex}, z_max{z1};

  gsl_function funct;
  funct.function = &fluct_2d_pp_f;
  funct.params = &params;

  if (funct.function(z_min, funct.params) > 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

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

double fluct_2d_n_ex_f(double E, void * params) {
  struct fluct_2d_n_ex_s * s{static_cast<fluct_2d_n_ex_s*>(params)};

  double z0{fluct_2d_pp(E, s->chi_ex, s->mu_e, s->mu_h, s->sys)};

  if (std::isnan(z0)) { return 0; }

  return 1 / (std::exp(z0) - 1) / (1 + 2 * (s->sys.get_z1(E, s->mu_e + s->mu_h) - z0) * fluct_2d_I2_dz(z0, E, s->mu_e, s->mu_h, s->sys));
}

double fluct_2d_n_ex(double chi_ex, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};
  double dt[n_int] = {0};

  fluct_2d_n_ex_s s{chi_ex, mu_e, mu_h, sys};

  gsl_function integrand;

  integrand.function = &fluct_2d_n_ex_f;
  integrand.params = &s;

  constexpr uint32_t local_ws_size{(1<<4)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  auto t0{std::chrono::high_resolution_clock::now()};
  gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  dt[0] = (std::chrono::high_resolution_clock::now() - t0).count();

  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e), %.2f μs\n", i, result[i], error[i], 1e-3 * dt[i]);
  }
  printf("\n");
  */

  return 0.25 * M_1_PI * sum_result<n_int>(result);
}

double fluct_2d_n_sc_ftx(double x, void * params) {
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  std::complex<double> b{
    //std::complex<double>(s->t, M_PI) + 2.0 * fluct_2d_I2(x - s->mu_e - s->mu_h - s->chi_ex * std::exp(-s->t), s->sys.m_sigma * x, s->mu_e, s->mu_h, s->sys)
    std::complex<double>(s->t, M_PI) + 2.0 * fluct_2d_I2_p(std::exp(-s->t), x, s->chi_ex, s->mu_e, s->mu_h, s->sys)
  };

  //std::cout << x << ' ' << b << std::endl;

  b = 1.0 / b;

  return b.imag() / (std::exp(x - s->mu_e - s->mu_h - s->chi_ex * std::exp(-s->t)) - 1.0);
}

double fluct_2d_n_sc_fzx(double x, void * params) {
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  std::complex<double> b{
    //std::complex<double>(-std::log(s->z), M_PI) + 2.0 * fluct_2d_I2(x - s->mu_e - s->mu_h - s->chi_ex * s->z, s->sys.m_sigma * x, s->mu_e, s->mu_h, s->sys)
    std::complex<double>(- std::log(s->z), M_PI) + 2.0 * fluct_2d_I2_p(s->z, x, s->chi_ex, s->mu_e, s->mu_h, s->sys)
  };

  b = 1.0 / b;

  return b.imag() / s->z / (std::exp(x - s->mu_e - s->mu_h - s->chi_ex * s->z) - 1.0);
}

double fluct_2d_n_sc_it(double t, void * params) {
  /*
   * Computes the branch cut integral of the spectral function,
   * with the fluctuations contribution.
   *
   * For now we will neglect dI2/dmu.
   */
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};

  s->t = t;

  gsl_function integrand;

  integrand.function = &fluct_2d_n_sc_ftx;
  integrand.params = s;

  constexpr uint32_t local_ws_size{(1<<6)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, global_eps, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  /*
  printf("t: %f\n", t);
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc_it: %d: %e (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return sum_result<n_int>(result);
}

double fluct_2d_n_sc_iz(double z, void * params) {
  /*
   * Computes the branch cut integral of the spectral function,
   * with the fluctuations contribution.
   *
   * For now we will neglect dI2/dmu.
   */
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};

  s->z = z;

  gsl_function integrand;

  integrand.function = &fluct_2d_n_sc_fzx;
  integrand.params = s;

  constexpr uint32_t local_ws_size{(1<<5)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, global_eps, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  /*
  printf("z: %f\n", z);
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc_iz: %d: %e (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return sum_result<n_int>(result) / z;
}

double fluct_2d_n_sc_if(double z, void * params) {
  /*
   * Computes the branch cut integral of the spectral function,
   * with the fluctuations contribution.
   *
   * For now we will neglect dI2/dmu.
   */
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  constexpr uint32_t n_int{2};
  double result[n_int] = {0}, error[n_int] = {0};

  s->t = z;
  s->z = z + 1;

  gsl_function integrands[2];

  integrands[0].function = &fluct_2d_n_sc_ftx;
  integrands[0].params = s;

  integrands[1].function = &fluct_2d_n_sc_fzx;
  integrands[1].params = s;

  constexpr uint32_t local_ws_size{(1<<6)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  gsl_integration_qagiu(integrands, global_eps, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_qagiu(integrands + 1, global_eps, 0, global_eps, local_ws_size, ws, result + 1, error + 1);

  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc_if: %d: %e (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return result[0] + result[1] / s->z;
}

double fluct_2d_n_sc(double chi_ex, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes the branch cut integral of the spectral function,
   * with the fluctuations contribution.
   *
   * For now we will neglect dI2/dmu.
   */

  constexpr uint32_t n_int{2};
  double result[n_int] = {0}, error[n_int] = {0};

  fluct_2d_n_sc_s s{0, 0, chi_ex, mu_e, mu_h, sys};

  gsl_function integrand[n_int];

  integrand[0].function = &fluct_2d_n_sc_it;
  integrand[0].params = &s;

  integrand[1].function = &fluct_2d_n_sc_iz;
  integrand[1].params = &s;

  constexpr uint32_t local_ws_size{(1<<6)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_qagiu(integrand + 1, 1, 0, global_eps, local_ws_size, ws, result + 1, error + 1);
  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc: %d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return 0.125 * M_1_PI * M_1_PI * sys.m_sigma * sum_result<n_int>(result);
}

double fluct_2d_n_sc_v2(double chi_ex, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes the branch cut integral of the spectral function,
   * with the fluctuations contribution.
   *
   * For now we will neglect dI2/dmu.
   */

  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};

  fluct_2d_n_sc_s s{0, 0, chi_ex, mu_e, mu_h, sys};

  gsl_function integrand[n_int];

  integrand[0].function = &fluct_2d_n_sc_if;
  integrand[0].params = &s;

  constexpr uint32_t local_ws_size{(1<<6)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc: %d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return 0.125 * M_1_PI * M_1_PI * sys.m_sigma * sum_result<n_int>(result);
}

int fluct_2d_mu_f(const gsl_vector * x, void * params, gsl_vector * f) {
  constexpr uint32_t n_eq{2};
  // defined in fluct_2d_utils.h
  fluct_2d_mu_s * s{static_cast<fluct_2d_mu_s*>(params)};

  double u{gsl_vector_get(x, 0)};
  double v{gsl_vector_get(x, 1)};

  //
  printf("prev u: %f, v: %f\n", u, v);

  if (std::isnan(u) || std::isnan(v)) {
    return GSL_EDOM;
  }

  double ls{s->ls_max / (std::exp(-v) + 1)};
  double mu_e{
    ideal_2d_mu_v(wf_E<2, 2>(ls, s->sys) - log(1 + std::exp(u)), s->sys)
  };

  double mu_h{ideal_2d_mu_h(mu_e, s->sys)};
  double mu_t{mu_e + mu_h};
  double chi_ex{wf_E<2, 2>(ls, s->sys)};

  double n_id{ideal_2d_n(mu_e, s->sys.m_pe)};
  double new_ls{ideal_2d_ls_mu(mu_e, mu_h, s->sys)};

  double n_ex{fluct_2d_n_ex(chi_ex, mu_e, mu_h, s->sys)};
  double n_sc{fluct_2d_n_sc(chi_ex, mu_e, mu_h, s->sys)};
  //double n_sc{0};

  ///*
  printf("u: %f, v: %f\n", u, v);
  printf("mu_e: %f, mu_t: %.15f, dif: %e\n", mu_e, mu_t, chi_ex - mu_t);
  printf("new_ls: %e, chi_ex: %.15f\n", new_ls, chi_ex);
  printf("n_id: %e, n_ex: %e, n_sc: %e\n\n", n_id, n_ex, n_sc);
  //*/

  double yv[n_eq] = {0};
  yv[0] = - s->n + n_id + (chi_ex > mu_t ? n_ex + n_sc : 0);
  yv[1] = - new_ls + ls;
  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }
  return GSL_SUCCESS;
}

std::vector<double> fluct_2d_mu(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  fluct_2d_mu_s params_s{n, ideal_2d_ls(1e20, sys), sys};

  gsl_multiroot_function f = {&fluct_2d_mu_f, n_eq, &params_s};

  double init_u{0};
  double init_v{0};

  double x_init[n_eq] = {
    init_u,
    init_v
  };

  gsl_vector * x = gsl_vector_alloc(n_eq);

  for(size_t i = 0; i < n_eq; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fsolver_type * T = gsl_multiroot_fsolver_hybrids;
  gsl_multiroot_fsolver * s = gsl_multiroot_fsolver_alloc(T, n_eq);
  gsl_multiroot_fsolver_set(s, &f, x);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    //printf("\niter %d: %.3f, %.10f, %.10f\n", iter, n, gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));

    auto t0{std::chrono::high_resolution_clock::now()};
    status = gsl_multiroot_fsolver_iterate(s);
    double dt = (std::chrono::high_resolution_clock::now() - t0).count();

    ///*
    printf("dt: %f s\n", 1e-9 * dt);
    printf("Status: %s\n", gsl_strerror(status));
    printf("x: ");
    for(size_t i = 0; i < n_eq; i++) { printf("%f, ", gsl_vector_get(s->x, i)); }

    printf("f: ");
    for(size_t i = 0; i < n_eq; i++) { printf("%e, ", gsl_vector_get(s->f, i)); }
    printf("\n");
    //*/

    if (status) { break; }
    status = gsl_multiroot_test_residual(s->f, global_eps);
  }

  std::vector<double> r(n_eq);

  r[1] = params_s.ls_max / (std::exp(-gsl_vector_get(s->x, 1)) + 1);
  r[0] = ideal_2d_mu_v(- std::log(1 + std::exp(gsl_vector_get(s->x, 0))) + wf_E<2, 2>(r[1], sys), sys);

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return r;
}

