#include "plasmons.h"
#include "plasmons_utils.h"

std::vector<double> plasmon_green(double w, double k, double mu_e, double mu_h, double v_1, const system_data & sys, double delta) {
  std::complex<double> w_complex(w, delta);

  double E[2] = {
    sys.m_pe * k * k,
    sys.m_ph * k * k
  };

  std::complex<double> nu[4] = {
    - w_complex / E[0] - 1.0, - w_complex / E[0] + 1.0,
    - w_complex / E[1] - 1.0, - w_complex / E[1] + 1.0
  };

  std::complex<double> pi_screen_nofactor[2] = {
    - 2.0
    - nu[0] * std::sqrt(1.0 - 4.0 * mu_e / ( E[0] * nu[0] * nu[0] ))
    + nu[1] * std::sqrt(1.0 - 4.0 * mu_e / ( E[0] * nu[1] * nu[1] )),
    - 2.0
    - nu[2] * std::sqrt(1.0 - 4.0 * mu_h / ( E[1] * nu[2] * nu[2] ))
    + nu[3] * std::sqrt(1.0 - 4.0 * mu_h / ( E[1] * nu[3] * nu[3] ))
  };

  std::complex<double> green = 1.0 / (
    - sys.eps_r / v_1 * std::abs(k)
    + (
        pi_screen_nofactor[0] / sys.m_pe
      + pi_screen_nofactor[1] / sys.m_ph
    ) * 0.125 * M_1_PI
  );

  return {green.real(), green.imag()};
}

double plasmon_kmax_f(double k, void * params) {
  plasmon_kmax_s * s{static_cast<plasmon_kmax_s*>(params)};

  double w{s->sys.m_pe * k*k + 2.0 * std::sqrt(s->sys.m_pe * s->mu_e) * k};

  double E[2] = {
    s->sys.m_pe * k * k,
    s->sys.m_ph * k * k
  };

  double nu[4] = {
    - w / E[0] - 1.0, - w / E[0] + 1.0,
    - w / E[1] - 1.0, - w / E[1] + 1.0
  };

  double pi_screen_nofactor[2] = {
    - 2.0
    - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[0] * nu[0] ))
    /*
     * When m_e < m_h, the argument of the sqrt is zero (or very close),
     * so it has to be removed to avoid complex numbers because of
     * numerical precision.
     */
    /*+ nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[1] * nu[1] ))*/,
    - 2.0
    - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[2] * nu[2] ))
    + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[3] * nu[3] ))
  };

  double r{- s->sys.eps_r / s->v_1 * k
    + (
        pi_screen_nofactor[0] / s->sys.m_pe
        + pi_screen_nofactor[1] / s->sys.m_ph
    ) * 0.125 * M_1_PI
  };

  return r;
}

double plasmon_kmax(double mu_e, double mu_h, double v_1, const system_data & sys) {
  /*
   * Evaluates the Green's function at
   *   w = m_pe * k^2 + 2 * sqrt(mu_e * m_pe) * k
   * and searches for the point at which Ginv is zero.
   *
   * This point is when the plasmon pole disappears.
   * Assumes k > 1e-5.
   * TODO: what about k < 1e-5?
   *
   * First perform an exponential sweep to find the upper
   * bound.
   * TODO: improve this? Analytic upper bound?
   */

  if (v_1 < 1e-5) {
    return v_1;
  }

  struct plasmon_kmax_s s{mu_e, mu_h, v_1, sys};

  double z{0};
  double z_min{1e-5}, z_max{1.0};

  gsl_function funct;
  funct.function = &plasmon_kmax_f;
  funct.params = &s;

  /*
   * Expontential sweep
   */

  const uint32_t max_pow{10};
  double ginv_upper{0}, upper_bound{z_max};
  for (uint32_t ii = 1; ii <= max_pow; ii++) {
    ginv_upper = funct.function(upper_bound, funct.params);

    if (ginv_upper < 0) {
      z_max = upper_bound;
      break;
    } else if (ginv_upper == 0) {
      return z_max;
    } else {
      z_min = upper_bound;
      upper_bound = z_max * (1<<ii);
    }
  }

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(solver, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(solver);
    z = gsl_root_fsolver_root(solver);
    z_min = gsl_root_fsolver_x_lower(solver);
    z_max = gsl_root_fsolver_x_upper(solver);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(solver);
  return z;
}

double plasmon_wmax(double mu_e, double mu_h, double v_1, const system_data & sys) {
  double kmax{plasmon_kmax(mu_e, mu_h, v_1, sys)};
  return sys.m_pe * kmax*kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax;
}

double plasmon_disp_f(double w, void * params) {
  plasmon_disp_s * s{static_cast<plasmon_disp_s*>(params)};

  double E[2] = {
    s->sys.m_pe * s->k * s->k,
    s->sys.m_ph * s->k * s->k
  };

  double nu[4] = {
    - w / E[0] - 1.0, - w / E[0] + 1.0,
    - w / E[1] - 1.0, - w / E[1] + 1.0
  };

  double pi_screen_nofactor[2] = {
    - 2.0
    - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[0] * nu[0] ))
    + nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[1] * nu[1] )),
    - 2.0
    - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[2] * nu[2] ))
    + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[3] * nu[3] ))
  };

  double r{
    - s->sys.eps_r / s->v_1 * s->k
    + (
        pi_screen_nofactor[0] / s->sys.m_pe
        + pi_screen_nofactor[1] / s->sys.m_ph
    ) * 0.125 * M_1_PI
  };

  return r;
}

template <bool check_bounds>
double plasmon_disp_tmpl(double k, double mu_e, double mu_h, double v_1, const system_data & sys, double kmax = 0) {
  if (k == 0) {
    return 0;
  } else if (k < 0) {
    k = -k;
  }

  if constexpr(check_bounds) {
    kmax = plasmon_kmax(mu_e, mu_h, v_1, sys);

    if (k > kmax) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  double z{0};
  double z_min{sys.m_pe * k*k + 2 * std::sqrt(sys.m_pe * mu_e) * k},
         z_max{sys.m_pe * kmax*kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax};

  z_min *= 1 + 1e-5;

  struct plasmon_disp_s s{k, mu_e, mu_h, v_1, sys};

  gsl_function funct;
  funct.function = &plasmon_disp_f;
  funct.params = &s;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(solver, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(solver);
    z = gsl_root_fsolver_root(solver);
    z_min = gsl_root_fsolver_x_lower(solver);
    z_max = gsl_root_fsolver_x_upper(solver);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(solver);
  return z;
}

double plasmon_disp(double k, double mu_e, double mu_h, double v_1, const system_data & sys) {
  return plasmon_disp_tmpl<true>(k, mu_e, mu_h, v_1, sys);
}

double plasmon_disp_ncb(double k, double mu_e, double mu_h, double v_1, const system_data & sys, double kmax) {
  return plasmon_disp_tmpl<false>(k, mu_e, mu_h, v_1, sys, kmax);
}

double plasmon_disp_inv_f(double k, void * params) {
  plasmon_disp_inv_s * s{static_cast<plasmon_disp_inv_s*>(params)};

  double E[2] = {
    s->sys.m_pe * k * k,
    s->sys.m_ph * k * k
  };

  double nu[4] = {
    - s->w / E[0] - 1.0, - s->w / E[0] + 1.0,
    - s->w / E[1] - 1.0, - s->w / E[1] + 1.0
  };

  double pi_screen_nofactor[2] = {
    - 2.0
    - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[0] * nu[0] ))
    + nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[1] * nu[1] )),
    - 2.0
    - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[2] * nu[2] ))
    + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[3] * nu[3] ))
  };

  double r{
    - s->sys.eps_r / s->v_1 * k
    + (
        pi_screen_nofactor[0] / s->sys.m_pe
        + pi_screen_nofactor[1] / s->sys.m_ph
    ) * 0.125 * M_1_PI
  };

  return r;
}

template <bool check_bounds>
double plasmon_disp_inv_tmpl(double w, double mu_e, double mu_h, double v_1, const system_data & sys) {
  /*
   * Essentially same as plasmon_disp, but it computes k(w), the inverse
   * dispersion relation.
   */

  if (w == 0) {
    return 0;
  } else if (w < 1e-5) {
    return w;
  } else if (w < 0) {
    w = -w;
  }

  if constexpr(check_bounds) {
    double kmax{plasmon_kmax(mu_e, mu_h, v_1, sys)};
    double w_max{sys.m_pe * kmax*kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax};

    if (w > w_max) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  double z{0};
  double z_min{1e-5},
         z_max{(- std::sqrt(mu_e) + std::sqrt(mu_e + w)) / std::sqrt(sys.m_pe)};

  struct plasmon_disp_inv_s s{w, mu_e, mu_h, v_1, sys};

  gsl_function funct;
  funct.function = &plasmon_disp_inv_f;
  funct.params = &s;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(solver, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(solver);
    z = gsl_root_fsolver_root(solver);
    z_min = gsl_root_fsolver_x_lower(solver);
    z_max = gsl_root_fsolver_x_upper(solver);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
  }

  gsl_root_fsolver_free(solver);
  return z;
}

double plasmon_disp_inv(double w, double mu_e, double mu_h, double v_1, const system_data & sys) {
  return plasmon_disp_inv_tmpl<true>(w, mu_e, mu_h, v_1, sys);
}

double plasmon_disp_inv_ncb(double w, double mu_e, double mu_h, double v_1, const system_data & sys) {
  return plasmon_disp_inv_tmpl<false>(w, mu_e, mu_h, v_1, sys);
}

std::vector<double> plasmon_green_ksq_f(double k2, plasmon_potcoef_s * s) {
  std::complex<double> w_complex(s->w, s->delta);

  double E[2] = {
    s->sys.m_pe * k2,
    s->sys.m_ph * k2
  };

  std::complex<double> nu[4] = {
    - w_complex / E[0] - 1.0, - w_complex / E[0] + 1.0,
    - w_complex / E[1] - 1.0, - w_complex / E[1] + 1.0
  };

  std::complex<double> pi_screen_nofactor[2] = {
    - 2.0
    - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[0] * nu[0] ))
    + nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[1] * nu[1] )),
    - 2.0
    - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[2] * nu[2] ))
    + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / ( E[1] * nu[3] * nu[3] ))
  };

  std::complex<double> green = 1.0 / (
    - s->sys.eps_r / s->v_1 * std::sqrt(k2)
    + (
        pi_screen_nofactor[0] / s->sys.m_pe
      + pi_screen_nofactor[1] / s->sys.m_ph
    ) * 0.125 * M_1_PI
  );

  return {green.real(), green.imag()};
}

double plasmon_potcoef_fr(double th, void * params) {
  plasmon_potcoef_s * s{static_cast<plasmon_potcoef_s*>(params)};

  double k2{
    s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)
  };

  std::vector<double> green = plasmon_green_ksq_f(k2, s);

  return green[0];
}

double plasmon_potcoef_fi(double th, void * params) {
  plasmon_potcoef_s * s{static_cast<plasmon_potcoef_s*>(params)};

  double k2{
    s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)
  };

  std::vector<double> green = plasmon_green_ksq_f(k2, s);

  return green[1];
}

std::vector<double> plasmon_potcoef(const std::vector<double> & wkk, double mu_e, double mu_h, double v_1, const system_data & sys, double delta) {
  constexpr uint32_t n_int{2};
  double result[n_int] = {0}, error[n_int] = {0};

  /*
   * wkk -> (w, k1, k2)
   */

  double k2{wkk[2]};
  if (std::abs(std::abs(wkk[1]/wkk[2]) - 1.0) < 1e-4) {
    k2 *= 1.0 + 1e-5;
  }

  plasmon_potcoef_s s{wkk[0], wkk[1], k2, mu_e, mu_h, v_1, sys, delta};

  gsl_function integrands[n_int];

  integrands[0].function = &plasmon_potcoef_fr;
  integrands[0].params = &s;

  integrands[1].function = &plasmon_potcoef_fi;
  integrands[1].params = &s;

  constexpr uint32_t local_ws_size{(1<<9)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qags(integrands, M_PI, 0, global_eps, 0, local_ws_size, ws, result, error);
  gsl_integration_qags(integrands + 1, 0, M_PI, global_eps, 0, local_ws_size, ws, result + 1, error + 1);
  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return {result[0] / M_PI, result[1] / M_PI};
}

std::vector<double> plasmon_sysmatelem(const std::vector<uint32_t> & ids, const std::vector<double> & wkwk, double dw, double dk, const std::complex<double> & z, const std::complex<double> & elem, const system_data & sys) {
  /*
   * ids[4] = {i, j, k, l}
   */

  std::complex<double> G0[2] = {
    z - sys.m_pe * std::pow(wkwk[1], 2),
    z - sys.m_ph * std::pow(wkwk[1], 2)
  };

  std::complex<double> result[2] = {
    std::sqrt(wkwk[3] / wkwk[1])
    * (
      static_cast<double>(kron(ids[0], ids[2]) * kron(ids[1], ids[3])) - dw * dk * elem / G0[0]
    ),
    std::sqrt(wkwk[3] / wkwk[1])
    * (
      static_cast<double>(kron(ids[0], ids[2]) * kron(ids[1], ids[3])) - dw * dk * elem / G0[1]
    )
  };

  return {result[0].real(), result[0].imag(), result[1].real(), result[1].imag()};
}

