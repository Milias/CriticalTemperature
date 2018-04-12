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
  if (exp_val > 9) { return std::numeric_limits<double>::quiet_NaN(); }

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
  gsl_integration_qagiu(&integrand2, 1 / chi_ex, 0, global_eps, local_ws_size, ws, result + 1, error + 1);

  gsl_integration_workspace_free(ws);

  return sys.m_sigma * 0.25 * M_1_PI * sum_result<n_int>(result);
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
  return 2 * M_PI / std::sqrt(
      (1 - std::exp(-M_PI * sys.m_pe * n_id)) / sys.m_pe
    + (1 - std::exp(-M_PI * sys.m_ph * n_id)) / sys.m_ph
  );
}

double analytic_2d_a_ls(double ls, const system_data & sys) {
  /*
   * NOTE: the scattering lengths here, like a0 and a1,
   * are not one over the scattering length as in the
   * rest of the program, but directly the dimensionless
   * scattering length.
   *
   * TODO: This is not valid in 2D!
   */
  state y{{0.0, 1.0}};
  double x0{1e-10}, x1{1<<8};
  double a0{x1}, a1;

  controlled_stepper_type controlled_stepper;
  wf_c<state, 0, 2> wf{ls, sys};

  for(uint8_t i = 0; i < max_iter; i++) {
    integrate_adaptive(controlled_stepper, wf, y, x0, x1, global_eps);

    a1 = x1 - y[0] / y[1];

    if (2 * std::abs(a1 - a0) / (a0 + a1) > global_eps) {
      x0 = x1;
      x1 *= 2;

      a0 = a1;
    } else {
      break;
    }
  }

  return -1.0 / a1;
}

int analytic_2d_mu_f(const gsl_vector * x, void * params, gsl_vector * f) {
  constexpr uint32_t n_eq{2};
  // defined in analytic_2d_utils.h
  analytic_2d_mu_s * s{static_cast<analytic_2d_mu_s*>(params)};

  double mu_e{gsl_vector_get(x, 0)};
  double ls{gsl_vector_get(x, 1)};

  double mu_h{ideal_2d_mu_h(mu_e, s->sys)};
  double mu_t{mu_e + mu_h};

  //printf("mu_e: %f, mu_t: %f\n", mu_e, mu_t);

  if (mu_t >= 0) {
    mu_t = -global_eps;
    mu_e = ideal_2d_mu_v(mu_t, s->sys);
  }

  double n_id{ideal_2d_n(mu_e, s->sys.m_pe)};
  double new_ls{ideal_2d_ls(n_id, s->sys)};
  double chi_ex{wf_E<0, 2>(ls, s->sys)};

  //printf("ls: %f, new_a: %f, max_a: %f\n", ls, new_a, std::sqrt(-2 * mu_t) * std::exp(M_EULER));

  double n_ex{analytic_2d_n_ex(mu_t, chi_ex, s->sys)};
  double n_sc{analytic_2d_n_sc(mu_t, chi_ex, s->sys)};

  //printf("n_id: %f, n_ex: %f, n_sc: %f\n", n_id, n_ex, n_sc);

  double yv[n_eq] = {0};
  yv[0] = - s->n + n_id + n_ex + n_sc;
  yv[1] = - ls + new_ls;
  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }

  return GSL_SUCCESS;
}

std::vector<double> analytic_2d_mu(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  analytic_2d_mu_s params_s{n, sys};

  gsl_multiroot_function f = {&analytic_2d_mu_f, n_eq, &params_s};

  //double init_mu{ideal_2d_mu(n, sys)};
  double init_mu{ideal_2d_mu_v(-1, sys)};
  //double mu_t{init_mu + ideal_2d_mu_h(init_mu, sys)};

  double x_init[n_eq] = {
    init_mu,
    ideal_2d_ls(n, sys)
  };

  //printf("first: %.3f, %.10f, %.10f, %.10f\n", n, x_init[0], x_init[1], mu_t);

  gsl_vector * x = gsl_vector_alloc(n_eq);

  for(size_t i = 0; i < n_eq; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fsolver_type * T = gsl_multiroot_fsolver_hybrids;
  gsl_multiroot_fsolver * s = gsl_multiroot_fsolver_alloc(T, n_eq);
  gsl_multiroot_fsolver_set(s, &f, x);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    //printf("iter %d: %.3f, %.10f, %.10f\n", iter, n, gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));

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

  for(size_t i = 0; i < n_eq; i++) { r[i] = gsl_vector_get(s->x, i); }

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return r;
}

