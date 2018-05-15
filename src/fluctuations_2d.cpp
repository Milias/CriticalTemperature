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

std::vector<double> fluct_2d_I2_b(double z, double E, double mu_e, double mu_h, double m) {
  /*
   * Computes bounds for the integral I2 when z > z1.
   * If z < z1 it returns {NaN, NaN}.
   *
   * The argument "m" is the ratio m_+ / m_i, and
   * therefore if m = m_pe, 1 - m = m_ph.
   */

  double z_min{(m - 1) * m * E + mu_e + mu_h + z};

  if (std::abs(z_min) < 1e-10) {
    return {
      m*m * E,
      m*m * E
    };
  }

  if (z_min < 0) {
    return {
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()
    };
  }

  return {
    z_min + m*m * E - 2 * m * std::sqrt(E * z_min),
    z_min + m*m * E + 2 * m * std::sqrt(E * z_min)
  };
}

double fluct_2d_I2_fe(double x, void * params) {
  /*
   * Integrand of I2.
   * Electron contribution.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_s * s{static_cast<fluct_2d_I2_s*>(params)};

  return fluct_2d_I2_f2e(x, s->z, s->E, s->mu_e, s->mu_h, s->sys);
}

double fluct_2d_I2_fh(double x, void * params) {
  /*
   * Integrand of I2.
   * Hole contribution.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_s * s{static_cast<fluct_2d_I2_s*>(params)};

  return fluct_2d_I2_f2h(x, s->z, s->E, s->mu_e, s->mu_h, s->sys);
}

double fluct_2d_I2_fce(double x, void * params) {
  /*
   * Integrand of I2, for Cauchy PV.
   * Electron contribution.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_s * s{static_cast<fluct_2d_I2_s*>(params)};

  return 1 / (std::exp((s->sys.m_pe * x - s->mu_e) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_fch(double x, void * params) {
  /*
   * Integrand of I2, for Cauchy PV.
   * Hole contribution.
   */

  // Defined in "fluctuations_2d_utils.h".
  fluct_2d_I2_s * s{static_cast<fluct_2d_I2_s*>(params)};

  return 1 / (std::exp((s->sys.m_ph * x - s->mu_h) * 0.25 * M_1_PI) + 1);
}

double fluct_2d_I2_zi(uint32_t s, double z, double mu_e, double mu_h, const system_data & sys) {
  /*
   * When E == 0 the imaginary part can be exactly computed.
   * s -> 0: Electron contribution.
   * s -> 1: Hole contribution.
   */

  if (s) {
    return 1 / (std::exp((sys.m_ph * (z + mu_e + mu_h) - mu_h) * 0.25 * M_1_PI) + 1);
  } else {
    return 1 / (std::exp((sys.m_pe * (z + mu_e + mu_h) - mu_e) * 0.25 * M_1_PI) + 1);
  }
}

std::complex<double> fluct_2d_I2(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes I2.
   * If z < z1 then it is real, otherwise it will have an imaginary part.
   */
  constexpr uint32_t n_spc{2}; // Number of species (electrons and holes = 2).
  constexpr uint32_t n_int{3 * n_spc};
  double result[n_int] = {0}, error[n_int] = {0};
  //size_t neval[n_int] = {0};

  fluct_2d_I2_s s{z, E, mu_e, mu_h, sys};

  /*
   * Two different integrands:
   * 0 -> Electrons.
   * 1 -> Holes.
   * 2 -> Cauchy electrons.
   * 3 -> Cauchy holes.
   */
  gsl_function integrands[2 * n_spc];

  integrands[0].function = &fluct_2d_I2_fe;
  integrands[0].params = &s;

  integrands[1].function = &fluct_2d_I2_fh;
  integrands[1].params = &s;

  integrands[2].function = &fluct_2d_I2_fce;
  integrands[2].params = &s;

  integrands[3].function = &fluct_2d_I2_fch;
  integrands[3].params = &s;

  constexpr uint32_t local_ws_size{(1<<8)};

  std::vector<double> bounds[n_spc] = {
    fluct_2d_I2_b(z, E, mu_e, mu_h, sys.m_ph),
    fluct_2d_I2_b(z, E, mu_e, mu_h, sys.m_pe)
  };

  //printf("bounds: (%f, %f), (%f, %f)\n", bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]);

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  for (uint32_t i = 0; i < n_spc; i++) {
    if (std::isnan(bounds[i][0])) {
      /*
       * If isnan(bounds[0]) it means that we are below the
       * branch cut, and therefore we only need to compute
       * the real part.
       */

      gsl_integration_qagiu(integrands + i, 0, 0, global_eps, local_ws_size, ws, result + i, error + i);

    } else if (bounds[i][0] == bounds[i][1]) {
      /*
       * If both bounds are equal it means E == 0, and
       * therefore we can easily compute the imaginary
       * part.
       *
       * In this case there is only one pole, therefore
       * we can compute the real part as the Cauchy
       * principal value.
       */
      if (bounds[i][0] > 0) {
        gsl_integration_qagiu(integrands + i, 2 * bounds[i][1], 0, global_eps, local_ws_size, ws, result + i, error + i);
        gsl_integration_qawc(integrands + n_spc + i, 0, 2 * bounds[i][1], bounds[i][1], 0, global_eps, local_ws_size, ws, result + n_spc + i, error + n_spc + i);
      } else {
        /*
         * This means E == 0.
         */
        result[i] = -std::numeric_limits<double>::infinity();
      }

      result[2 * n_spc + i] = fluct_2d_I2_zi(i, z, mu_e, mu_h, sys);

    } else {
      /*
       * Finally, the most general case.
       */

      //printf("bounds [%d]: (%f, %f)\n", i, bounds[i][0], bounds[i][1]);

      gsl_integration_qagiu(integrands + i, bounds[i][1] * (1 + global_eps), 0, global_eps, local_ws_size, ws, result + i, error + i);

      gsl_integration_qags(integrands + i, 0, bounds[i][0] * (1 - global_eps), 0, global_eps, local_ws_size, ws, result + n_spc + i, error + n_spc + i);
      //gsl_integration_qag(integrands + i, 0, bounds[i][0] * (1 - global_eps), 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result + n_spc + i, error + n_spc + i);

      //gsl_integration_qng(integrands + i, 0, bounds[i][0] * (1 - global_eps), 0, global_eps, result + n_spc + i, error + n_spc + i, neval + n_spc + i);

      gsl_integration_qags(integrands + i, bounds[i][0] * (1 + global_eps), bounds[i][1] * (1 - global_eps), 0, global_eps, local_ws_size, ws, result + 2 * n_spc + i, error + 2 * n_spc + i);
      //gsl_integration_qag(integrands + i, bounds[i][0] * (1 + global_eps), bounds[i][1] * (1 - global_eps), 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result + 2 * n_spc + i, error + 2 * n_spc + i);

      //gsl_integration_qng(integrands + i, bounds[i][0] + global_eps, bounds[i][1] - global_eps, 0, global_eps, result + 2 * n_spc + i, error + 2 * n_spc + i, neval + 2 * n_spc + i);

      if (std::isnan(result[i + 2 * n_spc])) {
        result[i + 2 * n_spc] = 0;
      }
    }
  }

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  gsl_integration_workspace_free(ws);

  return {
    // Re(I2_e)_x^inf + Re(I2_h)_x^inf + Re(I2_e)_0^x + Re(I2_h)_0^x
    0.25 * M_1_PI * (result[0] + result[1] + result[2] + result[3]),
    // Im(I2_e) + Im(I2_h)
    0.125 * M_1_PI * (result[4] + result[5])
  };
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
    gsl_integration_qagiu(integrands, 0, 0, global_eps, local_ws_size, ws, result, error);

    // 0 < x < x_-.
    gsl_integration_qag(integrands + 1, 0, 1, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result + 1, error + 1);

    // x_- < x < x_+.
    gsl_integration_qag(integrands + 2, 0, M_PI, 0, global_eps, local_ws_size, GSL_INTEG_GAUSS31, ws, result + 2, error + 2);
  } else {
    // In this case, s.x_0e == s.x_0h.
    gsl_integration_qagiu(integrands + 4, 2 * s.x_0e, 0, global_eps, local_ws_size, ws, result, error);
    gsl_integration_qawc(integrands + 3, 0, 2 * s.x_0e, s.x_0e, 0, global_eps, local_ws_size, ws, result + 1, error + 1);
    result[2] = fluct_2d_I2_p_i(&s);
  }

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  gsl_integration_workspace_free(ws);

  return {
    0.25 * M_1_PI * (-result[0] + result[1]),
    0.125 * M_1_PI * result[2]
  };
}

double fluct_2d_pp_f(double z, void * params) {
  struct fluct_2d_pp_s * s{static_cast<fluct_2d_pp_s*>(params)};

  return z - s->z1 - s->chi_ex * std::exp(2 * fluct_2d_I2(z, s->E, s->mu_e, s->mu_h, s->sys).real());
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

  return 1 / (std::exp(z0) - 1);
}

double fluct_2d_n_ex(double chi_ex, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{1};
  double result[n_int] = {0}, error[n_int] = {0};

  fluct_2d_n_ex_s s{chi_ex, mu_e, mu_h, sys};

  gsl_function integrand;

  integrand.function = &fluct_2d_n_ex_f;
  integrand.params = &s;

  constexpr uint32_t local_ws_size{(1<<4)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  return 0.25 * M_1_PI * sum_result<n_int>(result);
}

double fluct_2d_n_sc_ftx(double x, void * params) {
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  std::complex<double> b{
    std::complex<double>(s->t, M_PI) + 2.0 * fluct_2d_I2(x - s->mu_e - s->mu_h - s->chi_ex * std::exp(-s->t), s->sys.m_sigma * x, s->mu_e, s->mu_h, s->sys)
  };

  //std::cout << x << ' ' << b << std::endl;

  b = 1.0 / b;

  return b.imag() / (std::exp(x) - 1.0);
}

double fluct_2d_n_sc_fzx(double x, void * params) {
  struct fluct_2d_n_sc_s * s{static_cast<fluct_2d_n_sc_s*>(params)};

  std::complex<double> b{
    std::complex<double>(-std::log(s->z), M_PI) + 2.0 * fluct_2d_I2(x - s->mu_e - s->mu_h - s->chi_ex * s->z, s->sys.m_sigma * x, s->mu_e, s->mu_h, s->sys)
  };

  b = 1.0 / b;

  return b.imag() / s->z / (std::exp(x) - 1.0);
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

  constexpr uint32_t local_ws_size{(1<<7)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, global_eps, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  ///*
  printf("t: %f\n", t);
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc_it: %d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  //*/

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

  constexpr uint32_t local_ws_size{(1<<7)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, global_eps, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  ///*
  printf("z: %f\n", z);
  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc_iz: %d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  //*/

  return sum_result<n_int>(result) / z;
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

  constexpr uint32_t local_ws_size{(1<<5)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_qagiu(integrand + 1, 1, 0, global_eps, local_ws_size, ws, result + 1, error + 1);
  gsl_integration_workspace_free(ws);

  for (uint32_t i = 0; i < n_int; i++) {
    printf("fluct_2d_n_sc: %d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");

  return 0.25 * M_1_PI * M_1_PI * sys.m_sigma * sum_result<n_int>(result);
}

