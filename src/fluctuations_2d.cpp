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
    return M_PI / (std::exp((sys.m_ph * (z + mu_e + mu_h) - mu_h) * 0.25 * M_1_PI) + 1);
  } else {
    return M_PI / (std::exp((sys.m_pe * (z + mu_e + mu_h) - mu_e) * 0.25 * M_1_PI) + 1);
  }
}

std::complex<double> fluct_2d_I2(double z, double E, double mu_e, double mu_h, const system_data & sys) {
  /*
   * Computes I2.
   * If z < z1 then it is real, otherwise it will have an imaginary part.
   */
  constexpr uint32_t n_int{6};
  constexpr uint32_t n_spc{2}; // Number of species (electrons and holes = 2).
  double result[n_int] = {0}, error[n_int] = {0};

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

  constexpr uint32_t local_ws_size{(1<<6)};

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

      gsl_integration_qagiu(integrands + i, 2 * bounds[i][1], 0, global_eps, local_ws_size, ws, result + i, error + i);
      gsl_integration_qawc(integrands + n_spc + i, 0, 2 * bounds[i][1], bounds[i][1], 0, global_eps, local_ws_size, ws, result + n_spc + i, error + n_spc + i);

      result[2 * n_spc + i] = fluct_2d_I2_zi(i, z, mu_e, mu_h, sys);

    } else {
      /*
       * Finally, the most general case.
       */

      //printf("bounds [%d]: (%f, %f)\n", i, bounds[i][0], bounds[i][1]);

      gsl_integration_qagiu(integrands + i, bounds[i][1], 0, global_eps, local_ws_size, ws, result + i, error + i);
      gsl_integration_qags(integrands + i, 0, bounds[i][0], 0, global_eps, local_ws_size, ws, result + n_spc + i, error + n_spc + i);

      gsl_integration_qags(integrands + i, bounds[i][0], bounds[i][1], 0, global_eps, local_ws_size, ws, result + 2 * n_spc + i, error + 2 * n_spc + i);
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
    result[0] + result[1] + result[2] + result[3],
    // Im(I2_e) + Im(I2_h)
    result[4] + result[5]
  };
}

