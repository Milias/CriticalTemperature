#include "analytic.h"
#include "analytic_utils.h"

/*** Density ***/

double ideal_n(double mu_i, double m_pi) {
  return 2 * std::pow(m_pi, -1.5) * polylogExpM(1.5, mu_i / (4 * M_PI));
}

double analytic_n_ex(double mu_t, double a, const system_data & sys) {
  if (a < 0) { return 0; }
  return 4 * std::pow(sys.m_sigma, 1.5) * polylogExp(1.5, (mu_t + a*a / 4) / (4 * M_PI));
}

double analytic_n_sc_f(double y, void * params) {
  // defined in analytic_utils.h
  analytic_n_sc_s * s{static_cast<analytic_n_sc_s*>(params)};
  return polylogExp(1.5, s->mu_t / (4 * M_PI) - y*y) / (y*y + s->a*s->a / (16 * M_PI));
}

double analytic_n_sc(double mu_t, double a, const system_data & sys) {
  constexpr uint32_t n_int{1};
  constexpr uint32_t local_ws_size{1<<4};
  double result[n_int] = {0}, error[n_int] = {0};

  // defined in analytic_utils.h
  analytic_n_sc_s params_s{mu_t, a};

  gsl_function integrand;
  integrand.function = analytic_n_sc_f;
  integrand.params = &params_s;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, 0, 0, global_eps, local_ws_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  return - a * std::pow(sys.m_sigma / M_PI, 1.5) * result[0];
}

/*** Chemical potential ***/

double ideal_mu(double n, double m_pi) {
  return 4 * M_PI * invPolylogExpM(1.5, 0.25 * n * std::pow(m_pi, 1.5));
}

double ideal_mu_dn(double n, double m_pi) {
  return derivative_c2(&ideal_mu, n, n * 1e-6, m_pi);
}

double ideal_mu_h(double mu_e, const system_data & sys) {
  return 4 * M_PI * invPolylogExpM(1.5, std::pow(sys.m_ph / sys.m_pe, 1.5) * polylogExpM(1.5, mu_e / (4 * M_PI)));
}

double ideal_mu_v_f(double mu_e, void * params) {
  // defined in analytic_utils.h
  ideal_mu_v_s * s{static_cast<ideal_mu_v_s*>(params)};
  return mu_e + ideal_mu_h(mu_e, s->sys) - s->v;
}

double ideal_mu_v_df(double mu_e, void * params) {
  // defined in analytic_utils.h
  ideal_mu_v_s * s{static_cast<ideal_mu_v_s*>(params)};
  return 1 + derivative_c2(&ideal_mu_h, mu_e, global_eps * mu_e, s->sys);
}

void ideal_mu_v_fdf(double z, void * params, double * f, double * df) {
  f[0] = ideal_mu_v_f(z, params);
  df[0] = ideal_mu_v_df(z, params);
}

double ideal_mu_v(double v, double mu_0, const system_data & sys) {
  /*
    Solves the equation mu_e + mu_h == v for mu_e,
    assuming n_id,e == n_id,h.
  */

  // defined in analytic_utils.h
  ideal_mu_v_s params_s{v, sys};
  double z0, z{mu_0};

  gsl_function_fdf funct;
  funct.f = &ideal_mu_v_f;
  funct.df = &ideal_mu_v_df;
  funct.fdf = &ideal_mu_v_fdf;
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

double ideal_ls(double n_id, const system_data & sys) {
  double lambda_i[2] = {ideal_mu_dn(n_id, sys.m_pe), ideal_mu_dn(n_id, sys.m_ph)};
  return 1.0 / (1.0 / std::sqrt(std::abs(lambda_i[0])) + 1.0 / std::sqrt(std::abs(lambda_i[1])));
}

double analytic_a_ls(double ls, const system_data & sys) {
  typedef std::array<double, 2> state;
  typedef boost::numeric::odeint::runge_kutta_cash_karp54<state> error_stepper_type;
  typedef boost::numeric::odeint::controlled_runge_kutta<error_stepper_type> controlled_stepper_type;

  /*
   * NOTE: the scattering lengths here, like a0 and a1,
   * are not one over the scattering length as in the
   * rest of the program, but directly the dimensionless
   * scattering length.
   */
  state y{{0.0, 1.0}};
  double x0{1e-10}, x1{1<<6};
  double a0{x1}, a1;

  controlled_stepper_type controlled_stepper;
  analytic_a_n_s<state> wf{ls, sys};

  //printf("first: %.3f\n", wf.lambda_s);

  for(uint8_t i = 0; i < max_iter; i++) {
    integrate_adaptive(controlled_stepper, wf, y, x0, x1, x1 * 1e-3);

    a1 = x1 - y[0] / y[1];
    //printf("iter %d: %.8f\n", i, a1);

    if (2 * std::abs(a1 - a0) / (a0 + a1) > global_eps) {
      x0 = x1;
      x1 *= 2;

      a0 = a1;
    } else {
      break;
    }
  }

  return 1.0 / a1;
}

int analytic_mu_f(const gsl_vector * x, void * params, gsl_vector * f) {
  constexpr uint32_t n_eq{2};
  // defined in analytic_utils.h
  analytic_mu_s * s{static_cast<analytic_mu_s*>(params)};

  double mu_e{gsl_vector_get(x, 0)};
  double a{gsl_vector_get(x, 1)};

  double mu_h{ideal_mu_h(mu_e, s->sys)};
  double mu_t{mu_e + mu_h};

  double n_id{ideal_n(mu_e, s->sys.m_pe)};

  double ls{ideal_ls(n_id, s->sys)};

  double yv[n_eq] = {0};
  yv[0] = - s->n + 2 * n_id + analytic_n_ex(mu_t, a, s->sys) + analytic_n_sc(mu_t, a, s->sys);
  yv[1] = - a + analytic_a_ls(ls, s->sys);
  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }

  return GSL_SUCCESS;
}

std::vector<double> analytic_mu(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  analytic_mu_s params_s{n, sys};

  gsl_multiroot_function f = {&analytic_mu_f, n_eq, &params_s};

  /*
   * The initial guess is given by the minimum between
   * the ideal case and the purely excitonic case.
   */
  double x_init[n_eq] = {
    std::min(
      ideal_mu(n, sys.m_pe),
      ideal_mu_v(4 * M_PI * invPolylogExp(1.5, 0.25 * std::pow(sys.m_sigma, -1.5) * n), -1, sys)
    ),
    -100
  };

  //printf("first: %.3f, %.10f\n", n, x_init[0]);

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
  }

  std::vector<double> r(n_eq);

  for(size_t i = 0; i < n_eq; i++) { r[i] = gsl_vector_get(s->x, i); }

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return r;
}

