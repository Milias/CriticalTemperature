#include "plasmons.h"
#include "plasmons_utils.h"

std::vector<double> plasmon_green(double w, double k, double mu_e, double mu_h, const system_data & sys, double delta) {
  std::complex<double> w_complex(w, delta);

  double E[2] = {
    sys.m_pe * k * k,
    sys.m_ph * k * k
  };

  std::complex<double> nu[4] = {
    - w_complex + E[0], - w_complex - E[0],
    - w_complex + E[1], - w_complex - E[1]
  };

  std::complex<double> pi_screen_nofactor[2] = {
    - 2.0 * E[0]
    - nu[1] * std::sqrt(1.0 - 4.0 * mu_e * E[0] / ( nu[1] * nu[1] ))
    + nu[0] * std::sqrt(1.0 - 4.0 * mu_e * E[0] / ( nu[0] * nu[0] )),
    - 2.0 * E[1]
    - nu[3] * std::sqrt(1.0 - 4.0 * mu_h * E[1] / ( nu[3] * nu[3] ))
    + nu[2] * std::sqrt(1.0 - 4.0 * mu_h * E[1] / ( nu[2] * nu[2] ))
  };

  std::complex<double> green = 1.0 / (
    - 2 * sys.eps_r / sys.c_aEM * std::abs(k)
    + 0.5 * (
        pi_screen_nofactor[0] / E[0] / sys.m_pe
      + pi_screen_nofactor[1] / E[1] / sys.m_ph
    )
  );

  return {green.real(), green.imag()};
}

std::vector<double> plasmon_green_f(double k2, plasmon_potcoef_s * s) {
  std::complex<double> w_complex(s->w, s->delta);

  double E[2] = {
    s->sys.m_pe * k2,
    s->sys.m_ph * k2
  };

  std::complex<double> nu[4] = {
    - w_complex + E[0], - w_complex - E[0],
    - w_complex + E[1], - w_complex - E[1]
  };

  std::complex<double> pi_screen_nofactor[2] = {
    - 2.0 * E[0]
    - nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e * E[0] / ( nu[1] * nu[1] ))
    + nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e * E[0] / ( nu[0] * nu[0] )),
    - 2.0 * E[1]
    - nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h * E[1] / ( nu[3] * nu[3] ))
    + nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h * E[1] / ( nu[2] * nu[2] ))
  };

  std::complex<double> green = 1.0 / (
    - 2 * s->sys.eps_r / s->sys.c_aEM * std::sqrt(k2)
    + 0.5 * (
        pi_screen_nofactor[0] / E[0] / s->sys.m_pe
      + pi_screen_nofactor[1] / E[1] / s->sys.m_ph
    )
  );

  return {green.real(), green.imag()};
}

double plasmon_potcoef_fr(double th, void * params) {
  plasmon_potcoef_s * s{static_cast<plasmon_potcoef_s*>(params)};

  double k2{
    s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)
  };

  std::vector<double> green = plasmon_green_f(k2, s);

  return green[0];
}

double plasmon_potcoef_fi(double th, void * params) {
  plasmon_potcoef_s * s{static_cast<plasmon_potcoef_s*>(params)};

  double k2{
    s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)
  };

  std::vector<double> green = plasmon_green_f(k2, s);

  return green[1];
}

std::vector<double> plasmon_potcoef(double w, double k1, double k2, double mu_e, double mu_h, const system_data & sys) {
  constexpr uint32_t n_int{2};
  double result[n_int] = {0}, error[n_int] = {0};

  plasmon_potcoef_s s{w, k1, k2, mu_e, mu_h, sys};

  gsl_function integrands[n_int];

  integrands[0].function = &plasmon_potcoef_fr;
  integrands[0].params = &s;

  integrands[1].function = &plasmon_potcoef_fi;
  integrands[1].params = &s;

  constexpr uint32_t local_ws_size{(1<<8)};

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qag(integrands, M_PI, 0, global_eps, 0, local_ws_size, GSL_INTEG_GAUSS61, ws, result, error);
  gsl_integration_qag(integrands + 1, M_PI, 0, global_eps, 0, local_ws_size, GSL_INTEG_GAUSS61, ws, result + 1, error + 1);
  gsl_integration_workspace_free(ws);

  /*
  for (uint32_t i = 0; i < n_int; i++) {
    printf("%d: %f (%e)\n", i, result[i], error[i]);
  }
  printf("\n");
  */

  return {result[0] / M_PI, result[1] / M_PI};
}

std::vector<double> plasmon_sysmatelem(uint32_t i, uint32_t j, uint32_t k, uint32_t l, std::complex<double> z, double dk, std::complex<double> elem) {
  /*
   * ids[4] = {i, j, k, l}
   */

  std::complex<double> G0[2] = {
    z - std::pow(dk * i, 2),
    z - std::pow(dk * i, 2)
  };

  std::complex<double> result[2] = {
    dk * std::sqrt(i / k)
    * (
      static_cast<double>(kron(i, k) * kron(j, l)) - elem / G0[0]
    ),
    dk * std::sqrt(i / k)
    * (
      static_cast<double>(kron(i, k) * kron(j, l)) - elem / G0[1]
    )
  };

  return {result[0].real(), result[0].imag(), result[1].real(), result[1].imag()};
}

