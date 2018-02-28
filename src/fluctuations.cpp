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
  constexpr uint32_t local_ws_size{(1<<5)};

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
  fluct_T_i_s * s{static_cast<fluct_T_i_s*>(params)};

  if (s->E < global_eps) {
    return fluct_T_i_f_E0(x, s);
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

  constexpr uint32_t local_ws_size{(1<<5)};

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
  if (z > 0) {
    return derivative_b3(&fluct_T_i, z, z * 1e-3, E, mu_e, mu_h, sys);
  } else {
    return derivative_f3(&fluct_T_i, z, z * 1e-3, E, mu_e, mu_h, sys);
  }
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

std::complex<double> fluct_T_i_c(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);
