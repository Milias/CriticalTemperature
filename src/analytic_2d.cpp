#include "analytic_2d.h"
#include "analytic_2d_utils.h"

double mb_2d_iod(double n, double lambda_s, const system_data & sys) {
  double b_ex{wf_E<0, 2>(lambda_s, sys)};

  if (isnan(b_ex)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double v{2 * std::pow(2 * M_PI, 2) / (n * (sys.m_pe * sys.m_ph / sys.m_sigma)) * std::exp( b_ex / (4 * M_PI) )};

  return y_eq_s(v);
}

double mb_2d_iod_l(double n, double lambda_s, const system_data & sys) {
  double b_ex{wf_E<1, 2>(0, sys) + sys.delta_E / lambda_s};

  double v{2 * std::pow(2 * M_PI, 2) / (n * (sys.m_pe * sys.m_ph / sys.m_sigma)) * std::exp( b_ex / (4 * M_PI) )};

  return y_eq_s(v);
}

double ideal_2d_n(double mu_i, double m_pi) {
  return M_1_PI / m_pi * std::log(1 + std::exp(0.25 * mu_i * M_1_PI));
}

double analytic_2d_n_ex(double mu_t, double a, const system_data & sys) {
  double exp_val{mu_t + 2 * M_1_PI * a*a};

  if (exp_val > 0) {
    return 0;
  }

  return 0.25 * M_1_PI / (sys.m_pe * (1 + sys.m_pe)) * (-std::log(1 - std::exp(exp_val)));
}

double analytic_2d_n_sc_f(double y, void * params) {
  // defined in analytic_2d_utils.h
  analytic_2d_n_sc_s * s{static_cast<analytic_2d_n_sc_s*>(params)};
  return 0.5 * M_1_PI * s->a*s->a / (s->sys.m_pe * (1 + s->sys.m_pe) * ( M_PI * M_PI + std::pow(std::log(y), 2) )) * (-std::log(1 - std::exp(s->mu_t - 2 * M_1_PI * s->a*s->a * y)));
}

double analytic_2d_n_sc(double mu_t, double a, const system_data & sys) {
  constexpr uint32_t n_int{1};
  constexpr uint32_t local_ws_size{1<<4};
  double result[n_int] = {0}, error[n_int] = {0};

  // defined in analytic_2d_utils.h
  analytic_2d_n_sc_s params_s{mu_t, a, sys};

  gsl_function integrand;
  integrand.function = analytic_2d_n_sc_f;
  integrand.params = &params_s;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  return sum_result<n_int>(result);
}

double ideal_2d_mu_h(double mu_e, const system_data & sys) {
  return 4 * M_PI * std::log(std::exp(sys.m_ph / sys.m_pe * std::log(1 + std::exp(0.25 * mu_e * M_1_PI))) - 1);
}

