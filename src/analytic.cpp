#include "analytic.h"
#include "analytic_utils.h"

/*** Density ***/

double ideal_n(double mu_i, double m_pi) {
  return 2 * std::pow(m_pi, -1.5) * polylogExpM(1.5, mu_i / (4 * M_PI));
}

double ideal_dn_dmu(double mu_i, double m_pi) {
  return std::pow(m_pi, -1.5) * polylogExpM(0.5, mu_i / (4 * M_PI)) / (2 * M_PI);
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

template<bool N> double analytic_dn_f(double x1, double x2, const system_data & sys) {
  double mu_e, a;

  if constexpr(N) {
    mu_e = x1;
    a = x2;
  } else {
    a = x1;
    mu_e = x2;
  }

  double mu_t{mu_e + ideal_mu_h(mu_e, sys)};
  return analytic_n_ex(mu_t, a, sys) + analytic_n_sc(mu_t, a, sys);
}

double analytic_dn_dmu(double mu_e, double a, const system_data & sys) {
  return derivative_c2(&analytic_dn_f<true>, mu_e, mu_e * 1e-6, a, sys);
}

double analytic_dn_da(double mu_e, double a, const system_data & sys) {
  return derivative_c2(&analytic_dn_f<false>, a, a * 1e-6, mu_e, sys);
}

/*** Chemical potential ***/

double ideal_mu(double n, double m_pi) {
  return 4 * M_PI * invPolylogExpM(1.5, 0.5 * n * std::pow(m_pi, 1.5));
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

double ideal_mu_v(double v, const system_data & sys) {
  /*
    Solves the equation mu_e + mu_h == v for mu_e,
    assuming n_id,e == n_id,h.
  */

  double v_max{-1300};
  if (v < v_max) {
    return 0.5 * (v - v_max) + ideal_mu_v(v_max, sys);
  }

  // defined in analytic_utils.h
  ideal_mu_v_s params_s{v, sys};
  double z0, z{v > 0 ? sys.m_pe * v : 0.5 * v};

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
  /*
   * NOTE: the scattering lengths here, like a0 and a1,
   * are not one over the scattering length as in the
   * rest of the program, but directly the dimensionless
   * scattering length.
   *
   * "state", "error_stepper_type" and "controlled_stepper_type"
   * are defined in "analytic_utils.h".
   */
  state y{{0.0, 1.0}};
  double x0{1e-10}, x1{1<<6};
  double a0{x1}, a1;

  controlled_stepper_type controlled_stepper;
  wf_c<state> wf{ls, sys};

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
  double new_a{analytic_a_ls(ls, s->sys)};

  double yv[n_eq] = {0};
  yv[0] = - s->n + n_id + analytic_n_ex(mu_t, new_a, s->sys) + analytic_n_sc(mu_t, new_a, s->sys);
  yv[1] = - a + new_a;
  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }

  return GSL_SUCCESS;
}

std::vector<double> analytic_mu_f(double mu_e, double a, double n, const system_data & sys) {
  gsl_vector * x = gsl_vector_alloc(2);
  gsl_vector_set(x, 0, mu_e);
  gsl_vector_set(x, 1, a);

  analytic_mu_s params{n, sys};

  analytic_mu_f(x, &params, x);

  std::vector<double> r(2);
  for (uint32_t i = 0; i < 2; i++) { r[i] = gsl_vector_get(x, i); }

  gsl_vector_free(x);
  return r;
}

double analytic_mu_init_a(double n, double a, const system_data & sys) {
  return -0.25 * a*a + 4 * M_PI * invPolylogExp(1.5, 0.5 * std::pow(sys.m_sigma, -1.5) * n);
}

double analytic_mu_init_mu(double n, double a, const system_data & sys) {
  if (a > 0) {
    //return ideal_mu_v(analytic_mu_init_a(n, a, sys), sys);
    //return ideal_mu_v(-0.25 * a*a, sys);
    return std::min(ideal_mu(n, sys.m_pe), ideal_mu_v(analytic_mu_init_a(n, a, sys), sys));
    //return ideal_mu(n, sys.m_pe);
  } else {
    return ideal_mu(n, sys.m_pe);
  }
}

std::vector<double> analytic_mu(double n, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  analytic_mu_s params_s{n, sys};

  gsl_multiroot_function f = {&analytic_mu_f, n_eq, &params_s};

  double init_a{analytic_a_ls(ideal_ls(2*ideal_n(-300, sys.m_pe), sys), sys)};
  double x_init[n_eq] = {
    analytic_mu_init_mu(n, init_a, sys),
    init_a
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

template<uint32_t N, uint32_t M> double analytic_mu_df_nm(double mu_e, double a, const analytic_mu_s * s) {
  if constexpr(N == 0) {
    if constexpr(M == 0) {
      /*
       * dn_eq/dmu
       */
      return ideal_dn_dmu(mu_e, s->sys.m_pe) + analytic_dn_dmu(mu_e, a, s->sys);

    } else if constexpr(M == 1) {
      /*
       * dn_eq/da
       */
      return analytic_dn_da(mu_e, a, s->sys);

    }
  } else if constexpr(N == 1) {
    if constexpr(M == 0) {
      /*
       * da_eq/dmu
       */
      double n_id{ideal_n(mu_e, s->sys.m_pe)};
      double ls{ideal_ls(n_id, s->sys)};
      return derivative_c2(&analytic_a_ls, ls, ls * 1e-6, s->sys);

    } else if constexpr(M == 1) {
      /*
       * da_eq/da
       */
      return 0;

    }
  }
}

int analytic_mu_df(const gsl_vector * x, void * params, gsl_matrix * J) {
  // defined in analytic_utils.h
  analytic_mu_s * s{static_cast<analytic_mu_s*>(params)};

  double mu_e{gsl_vector_get(x, 0)};
  double a{gsl_vector_get(x, 1)};

  gsl_matrix_set(J, 0, 0, analytic_mu_df_nm<0, 0>(mu_e, a, s));
  gsl_matrix_set(J, 0, 1, analytic_mu_df_nm<0, 1>(mu_e, a, s));
  gsl_matrix_set(J, 1, 0, analytic_mu_df_nm<1, 0>(mu_e, a, s));
  gsl_matrix_set(J, 1, 1, analytic_mu_df_nm<1, 1>(mu_e, a, s));

  return GSL_SUCCESS;
}

int analytic_mu_fdf(const gsl_vector * x, void * params, gsl_vector * f, gsl_matrix * J) {
  analytic_mu_f(x, params, f);
  analytic_mu_df(x, params, J);

  return GSL_SUCCESS;
}

std::vector<double> analytic_mu_follow(double n, std::vector<double> x_init, const system_data & sys) {
  constexpr uint32_t n_eq{2};
  analytic_mu_s params_s{n, sys};

  gsl_multiroot_function_fdf f = {
    &analytic_mu_f,
    &analytic_mu_df,
    &analytic_mu_fdf,
    n_eq,
    &params_s
  };

  //printf("first: %.3f, %.10f\n", n, x_init[0]);

  gsl_vector * x = gsl_vector_alloc(n_eq);

  for(size_t i = 0; i < n_eq; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fdfsolver_type * T = gsl_multiroot_fdfsolver_hybridsj;
  gsl_multiroot_fdfsolver * s = gsl_multiroot_fdfsolver_alloc(T, n_eq);
  gsl_multiroot_fdfsolver_set(s, &f, x);

  std::vector<double> r;
  for(size_t i = 0; i < n_eq; i++) {
    r.push_back(gsl_vector_get(s->x, i));
    r.push_back(gsl_vector_get(s->f, i));
  }

  int status, iter;
  for (status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    //printf("iter %d: %.3f, %.10f, %.10f\n", iter, n, gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));
    status = gsl_multiroot_fdfsolver_iterate(s);
    for(size_t i = 0; i < n_eq; i++) {
      r.push_back(gsl_vector_get(s->x, i));
      r.push_back(gsl_vector_get(s->f, i));
    }
    if (status) { break; }
    status = gsl_multiroot_test_residual(s->f, global_eps);
    //status = gsl_multiroot_test_delta(s->dx, s->x, 0, global_eps);
  }

  //printf("STATUS: %s\n", gsl_strerror(status));

  for(size_t i = 0; i < n_eq; i++) {
    r.push_back(gsl_vector_get(s->x, i));
    r.push_back(gsl_vector_get(s->f, i));
  }

  gsl_multiroot_fdfsolver_free(s);
  gsl_vector_free(x);

  return r;
}

double mb_iod(double n, double lambda_s, const system_data & sys) {
  double b_ex{wf_E<0>(lambda_s, sys)};

  if (isnan(b_ex)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double v{2 * std::pow(2 * M_PI, 3) * std::pow(sys.m_pe + sys.m_ph, -1.5) / n * std::exp( - b_ex / (4 * M_PI) )};

  return y_eq_s(v);
}

double mb_iod_l(double n, double lambda_s, const system_data & sys) {
  double b_ex{sys.get_E_n<1>() + sys.delta_E / lambda_s};

  double v{2 * std::pow(2 * M_PI, 3) * std::pow(sys.m_pe + sys.m_ph, -1.5) / n * std::exp( - b_ex / (4 * M_PI) )};

  return y_eq_s(v);
}

