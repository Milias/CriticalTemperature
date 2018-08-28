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

double analytic_2d_n_ex(double mu_t, double chi_ex, const system_data & sys) {
  double exp_val{mu_t - chi_ex};
  //if (exp_val > 9) { return std::numeric_limits<double>::quiet_NaN(); }
  if (exp_val > 9) { return 0; }

  return 0.25 * M_1_PI * sys.m_sigma * (-std::log(1 - std::exp(exp_val)));
}

double analytic_2d_n_sc_f1(double y, void * params) {
  // defined in analytic_2d_utils.h
  analytic_2d_n_sc_s * s{static_cast<analytic_2d_n_sc_s*>(params)};

  return (-std::log(1 - std::exp(s->mu_t + s->chi_ex * std::exp(-y))))
    / ( M_PI * M_PI + y*y );
}

double analytic_2d_n_sc_f2(double y, void * params) {
  // defined in analytic_2d_utils.h
  analytic_2d_n_sc_s * s{static_cast<analytic_2d_n_sc_s*>(params)};

  return (-std::log(1 - std::exp(s->mu_t + s->chi_ex * y)))
    / ( M_PI * M_PI + std::pow(std::log(y), 2) ) / y;
}

double analytic_2d_n_sc(double mu_t, double chi_ex, const system_data & sys) {
  constexpr uint32_t n_int{2};
  constexpr uint32_t local_ws_size{1<<6};
  double result[n_int] = {0}, error[n_int] = {0};

  // defined in analytic_2d_utils.h
  analytic_2d_n_sc_s params_s{mu_t, chi_ex, sys};

  gsl_function integrand1;
  integrand1.function = analytic_2d_n_sc_f1;
  integrand1.params = &params_s;

  gsl_function integrand2;
  integrand2.function = analytic_2d_n_sc_f2;
  integrand2.params = &params_s;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);

  gsl_integration_qagiu(&integrand1, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_qagiu(&integrand2, -1 / chi_ex, 0, global_eps, local_ws_size, ws, result + 1, error + 1);

  gsl_integration_workspace_free(ws);

  printf("%e (%e), %e (%e)\n", result[0], error[0], result[1], error[1]);

  return - sys.m_sigma * 0.25 * M_1_PI * sum_result<n_int>(result);
}

double ideal_2d_n_dmu(double mu_i, double m_pi) {
  return 0.25 * M_1_PI * M_1_PI / ( m_pi * (1 + std::exp(-0.25 * M_1_PI * mu_i)));
}

double analytic_2d_n_ex_dmu(double mu_t, double chi_ex, const system_data & sys) {
  return sys.m_sigma * 0.25 * M_1_PI / (std::exp(chi_ex - mu_t) - 1);
}

double analytic_2d_n_sc_dmu(double mu_t, double chi_ex, const system_data & sys) {
  return derivative_b3(analytic_2d_n_sc, mu_t, 1e-6 * mu_t, chi_ex, sys);
}

double analytic_2d_n_ex_dchi(double mu_t, double chi_ex, const system_data & sys) {
  return sys.m_sigma * 0.25 * M_1_PI / ( 1 - std::exp(chi_ex - mu_t) );
}

double analytic_2d_n_sc_dchi_f(double chi_ex, double mu_t, const system_data & sys) {
  return analytic_2d_n_sc(mu_t, chi_ex, sys);
}

double analytic_2d_n_sc_dchi(double mu_t, double chi_ex, const system_data & sys) {
  return derivative_b3(analytic_2d_n_sc_dchi_f, chi_ex, 1e-6 * chi_ex, mu_t, sys);
}

double ideal_2d_mu(double n_id, const system_data & sys) {
  return 4 * M_PI * std::log(std::exp(n_id * M_PI * sys.m_pe) - 1);
}

double ideal_2d_mu_h(double mu_e, const system_data & sys) {
  return 4 * M_PI * std::log(std::exp(sys.m_ph / sys.m_pe * std::log(1 + std::exp(0.25 * mu_e * M_1_PI))) - 1);
}

double analytic_2d_mu_ex(double chi_ex, double n_ex, const system_data & sys) {
  return std::log(1 - std::exp(- 4 * M_PI / sys.m_sigma * n_ex)) + chi_ex;
}

double ideal_2d_mu_h_dmu(double mu_e, const system_data & sys) {
  return sys.m_ph / sys.m_pe
    / (
      (1 + std::exp(- 0.25 * M_1_PI * mu_e))
      * (1 - std::exp(- sys.m_ph / sys.m_pe * std::log(1 + std::exp(0.25 * M_1_PI * mu_e))))
    );
}

double ideal_2d_mu_v_f(double mu_e, void * params) {
  // defined in analytic_2d_utils.h
  ideal_2d_mu_v_s * s{static_cast<ideal_2d_mu_v_s*>(params)};
  return mu_e + ideal_2d_mu_h(mu_e, s->sys) - s->v;
}

double ideal_2d_mu_v_df(double mu_e, void * params) {
  // defined in analytic_2d_utils.h
  ideal_2d_mu_v_s * s{static_cast<ideal_2d_mu_v_s*>(params)};
  double m_ratio{s->sys.m_ph / s->sys.m_pe};

  return 1 + m_ratio
    / (
        (1 + std::exp(- 0.25 * M_1_PI * mu_e))
        * (1 - std::exp( - m_ratio * std::log(1 + std::exp(0.25 * M_1_PI * mu_e))))
      );
}

void ideal_2d_mu_v_fdf(double z, void * params, double * f, double * df) {
  f[0] = ideal_2d_mu_v_f(z, params);
  df[0] = ideal_2d_mu_v_df(z, params);
}

double ideal_2d_mu_v(double v, const system_data & sys) {
  /*
    Solves the equation mu_e + mu_h == v for mu_e,
    assuming n_id,e == n_id,h.
  */

  double v_max{-1300};
  if (v < v_max) {
    return 0.5 * (v - v_max) + ideal_2d_mu_v(v_max, sys);
  }

  // defined in analytic_2d_utils.h
  ideal_2d_mu_v_s params_s{v, sys};
  double z0, z{v > 0 ? sys.m_pe * v : 0.5 * v};

  gsl_function_fdf funct;
  funct.f = &ideal_2d_mu_v_f;
  funct.df = &ideal_2d_mu_v_df;
  funct.fdf = &ideal_2d_mu_v_fdf;
  funct.params = &params_s;

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

double ideal_2d_ls(double n_id, const system_data & sys) {
  return sys.c_aEM * std::sqrt(2 * sys.m_pT) * (
      (1 - std::exp(- M_PI * sys.m_pe * n_id)) / sys.m_pe
      + (1 - std::exp(- M_PI * sys.m_ph * n_id)) / sys.m_ph
    ) / (2 * M_PI * M_PI * sys.eps_r);
}

double ideal_2d_ls_mu(double mu_e, double mu_h, const system_data & sys) {
  return (
      1 / ((1 + std::exp(- 0.25 * M_1_PI * mu_e)) * sys.m_pe)
      + 1 / ((1 + std::exp(- 0.25 * M_1_PI * mu_h)) * sys.m_ph)
    ) / (2 * M_PI * M_PI * sys.eps_r / (sys.c_aEM * std::sqrt(2 * sys.m_pT)));
}

int analytic_2d_mu_f(const gsl_vector * x, void * params, gsl_vector * f) {
  constexpr uint32_t n_eq{2};
  // defined in analytic_2d_utils.h
  analytic_2d_mu_s * s{static_cast<analytic_2d_mu_s*>(params)};

  double u{gsl_vector_get(x, 0)};
  double v{gsl_vector_get(x, 1)};

  if (std::isnan(u)) { u = 0; }
  if (std::isnan(v)) { v = 0; }

  double ls{s->ls_max / (std::exp(-v) + 1)};
  double mu_e{
    ideal_2d_mu_v(wf_E<2, 2>(ls, s->sys) - log(1 + std::exp(u)), s->sys)
  };

  double mu_h{ideal_2d_mu_h(mu_e, s->sys)};
  double mu_t{mu_e + mu_h};
  double chi_ex{wf_E<2, 2>(ls, s->sys)};

  double n_id{ideal_2d_n(mu_e, s->sys.m_pe)};
  double new_ls{ideal_2d_ls_mu(mu_e, mu_h, s->sys)};

  double n_ex{analytic_2d_n_ex(mu_t, chi_ex, s->sys)};
  double n_sc{analytic_2d_n_sc(mu_t, chi_ex, s->sys)};

  /*
  printf("u: %f, v: %f\n", u, v);
  printf("mu_e: %f, mu_t: %.15f, dif: %e\n", mu_e, mu_t, chi_ex - mu_t);
  printf("new_ls: %e, chi_ex: %.15f\n", new_ls, chi_ex);
  printf("n_id: %e, n_ex: %e, n_sc: %e\n\n", n_id, n_ex, n_sc);
  */

  double yv[n_eq] = {0};
  yv[0] = - s->n + n_id + (chi_ex > mu_t ? n_ex + n_sc : 0);
  yv[1] = - new_ls + ls;
  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }
  return GSL_SUCCESS;
}

std::vector<double> analytic_2d_mu(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  analytic_2d_mu_s params_s{n, ideal_2d_ls(1e20, sys), sys};

  gsl_multiroot_function f = {&analytic_2d_mu_f, n_eq, &params_s};

  double init_u{0};
  double init_v{0};

  double x_init[n_eq] = {
    init_u,
    init_v
  };

  //printf("first: %.3f, %.10f, %.10f, %.10f, chi_ex: %.2f\n", n, x_init[0], x_init[1], ideal_2d_mu_h(init_mu, sys), chi_ex);

  gsl_vector * x = gsl_vector_alloc(n_eq);

  for(size_t i = 0; i < n_eq; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fsolver_type * T = gsl_multiroot_fsolver_hybrids;
  gsl_multiroot_fsolver * s = gsl_multiroot_fsolver_alloc(T, n_eq);
  gsl_multiroot_fsolver_set(s, &f, x);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    //printf("\niter %d: %.3f, %.10f, %.10f\n", iter, n, gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));

    status = gsl_multiroot_fsolver_iterate(s);
    if (status) { break; }
    status = gsl_multiroot_test_residual(s->f, global_eps);

    /*
    printf("x: ");
    for(size_t i = 0; i < n_eq; i++) { printf("%f, ", gsl_vector_get(s->x, i)); }

    printf("f: ");
    for(size_t i = 0; i < n_eq; i++) { printf("%e, ", gsl_vector_get(s->f, i)); }
    printf("\n");
    */
  }

  std::vector<double> r(n_eq);

  r[1] = params_s.ls_max / (std::exp(-gsl_vector_get(s->x, 1)) + 1);
  r[0] = ideal_2d_mu_v(- std::log(1 + std::exp(gsl_vector_get(s->x, 0))) + wf_E<2, 2>(r[1], sys), sys);

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return r;
}

double analytic_2d_mu_f_py(double mu_e, double /*ls*/, double n, const system_data & sys) {
  double mu_h{ideal_2d_mu_h(mu_e, sys)};
  double mu_t{mu_e + mu_h};

  double n_id{ideal_2d_n(mu_e, sys.m_pe)};
  double new_ls{ideal_2d_ls_mu(mu_e, mu_h, sys)};
  double chi_ex{wf_E<2, 2>(new_ls, sys)};

  if (chi_ex > mu_t) {
    double n_ex{analytic_2d_n_ex(mu_t, chi_ex, sys)};
    double n_sc{analytic_2d_n_sc(mu_t, chi_ex, sys)};

    //return std::pow(- n + n_id + n_ex + n_sc, 2) + std::pow(- new_ls + ls, 2);
    return - n + n_id + n_ex + n_sc;
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double analytic_2d_mu_f_optim(const arma::vec & x, arma::vec * /*df*/, void * params) {
  // defined in analytic_2d_utils.h
  analytic_2d_mu_s * s{static_cast<analytic_2d_mu_s*>(params)};

  //std::cout << "x:\n" << x << std::endl;

  double u{x(0)};
  double v{x(1)};

  //printf("u: %.14f, v: %.14f\n", u, v);

  double ls{s->ls_max / (std::exp(-v) + 1)};
  double mu_e{
    ideal_2d_mu_v(wf_E<2, 2>(ls, s->sys) - std::exp(u), s->sys)
  };

  double mu_h{ideal_2d_mu_h(mu_e, s->sys)};
  double mu_t{mu_e + mu_h};
  double chi_ex{wf_E<2, 2>(ls, s->sys)};

  //printf("mu_e: %f, mu_t: %f, dif: %e\n", mu_e, mu_t, chi_ex - mu_t);

  //double new_ls{ideal_2d_ls(n_id, s->sys)};

  double n_id{ideal_2d_n(mu_e, s->sys.m_pe)};
  double new_ls{ideal_2d_ls_mu(mu_e, mu_h, s->sys)};

  //printf("new_ls: %f, chi_ex: %f\n", new_ls, chi_ex);

  double n_ex{analytic_2d_n_ex(mu_t, chi_ex, s->sys)};
  double n_sc{analytic_2d_n_sc(mu_t, chi_ex, s->sys)};

  //printf("n_id: %e, n_ex: %e, n_sc: %e\n", n_id, n_ex, n_sc);

  return std::pow(- s->n + n_id + n_ex + n_sc, 2) + std::pow(- new_ls + ls, 2);
}

std::vector<double> analytic_2d_mu_optim(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  analytic_2d_mu_s params_s{n, ideal_2d_ls(1e20, sys), sys};

  double x_init[n_eq] = {
    0,
    0
  };

  //printf("first: %.3e, %.10f, %.10f, %.10f, chi_ex: %.2f\n", n, x_init[0], x_init[1], low_mu_h, chi_ex);

  arma::vec x{x_init, n_eq};

  optim::algo_settings settings;
  settings.err_tol = global_eps;
  settings.iter_max = 1<<12;

  optim::nm(x, analytic_2d_mu_f_optim, &params_s, settings);

  printf("result: %e\n", analytic_2d_mu_f_optim(x, nullptr, &params_s));

  std::vector<double> r(n_eq);

  r[1] = params_s.ls_max / (std::exp(-x(1)) + 1);
  r[0] = ideal_2d_mu_v(- std::exp(x(0)) + wf_E<2, 2>(r[1], sys), sys);

  return r;
}

