#include "integrals.h"

/*** I1 ***/

double I1(double z2) {
  return - sqrt(z2);
}

double I1dmu(double z2) {
  return 1 / sqrt(z2);
}

/*** I2 ***/

double integrandI2part1(double x, void * params) {
  double * params_arr = (double *)params;
  double /*w,*/ E, mu, beta, r;
  //w = params_arr[0];
  E = params_arr[1];
  mu = params_arr[2];
  beta = params_arr[3];

  if (E < 1e-6) {
    r = sqrt(x) / (exp(beta * (x - mu)) + 1);
  } else {
    double E_ex = 0.25 * E - mu + x;
    double Ep = beta * (E_ex + sqrt( E * x )), Em = beta * (E_ex - sqrt( E * x ));

    r = sqrt(x) + (logExp_mpfr(Em) - logExp_mpfr(Ep)) / ( 2 * beta * sqrt(E) );
  }

  return r;
}

double integrandI2part2v2(double x, void * params) {
  double * params_arr = (double *)params;
  double /*w,*/ E, mu, beta, r;
  //w = params_arr[0];
  E = params_arr[1];
  mu = params_arr[2];
  beta = params_arr[3];

  if (E < 1e-6) {
    r = 1 / ((exp(beta * (x - mu)) + 1) * sqrt(x) );
  } else {
    double E_ex = 0.25 * E - mu + x;
    double Ep = beta * (E_ex + sqrt( E * x )), Em = beta * (E_ex - sqrt( E * x ));

    r = 1 / sqrt(x) + (logExp_mpfr(Em) - logExp_mpfr(Ep)) / ( 2 * beta * sqrt(E) * x);
  }

  return r;
}

double integrandI2part2(double x, void * params) {
  double * params_arr = (double *)params;

  return integrandI2part1(x, params) / (x - params_arr[4]);
}

double integralI2Real(double w, double E, double mu, double beta) {
  double result[2] = {0, 0}, error[] = {0, 0};
  double params_arr[] = {w, E, mu, beta, 0.5 * w - 0.25 * E + mu};

  gsl_function integrand_part2;
  integrand_part2.function = &integrandI2part2;
  integrand_part2.params = params_arr;

  size_t local_w_size = (1<<3) + (1<<2);

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_w_size);

  if (params_arr[4] < 0) {
    size_t neval;
    gsl_integration_qng(&integrand_part2, 0, E, 0, 1e-10, result, error, &neval);
    gsl_integration_qagiu(&integrand_part2, E, 0, error[0], local_w_size, ws, result + 1, error + 1);
  } else if (params_arr[4] == 0) {
    gsl_function integrand_part1;
    integrand_part1.function = &integrandI2part2v2;
    integrand_part1.params = params_arr;

    gsl_integration_qags(&integrand_part1, 0, 1, 0, 1e-10, local_w_size, ws, result, error);
    gsl_integration_qagiu(&integrand_part2, 1, 0, error[0], local_w_size, ws, result + 1, error + 1);
  } else {
    gsl_function integrand_part1;
    integrand_part1.function = &integrandI2part1;
    integrand_part1.params = params_arr;

    double x_max = 4 * fmax(1e-4, params_arr[4]);

    gsl_integration_qawc(&integrand_part1, 0, x_max, params_arr[4], 0, 1e-10, local_w_size, ws, result, error);
    gsl_integration_qagiu(&integrand_part2, 0, 0, error[0], local_w_size, ws, result + 1, error + 1);
  }

  gsl_integration_workspace_free(ws);

  //printf("integralI2Real: %.10f, %.10f, %.10f, %.10f, %.1e, %.1e\n", w, params_arr[4], result[0], result[1], error[0], error[1]);

  return 2 / M_PI * (result[0] + result[1]);
}

double integralI2Imag(double w, double E, double mu, double beta) {
  double params_arr[] = {w, E, mu, beta, 0.5 * w - 0.25 * E + mu};

  return 2 * integrandI2part1(params_arr[4], params_arr);
}

std::complex<double> I2(double w, double E, double mu, double beta) {
  return std::complex<double>(integralI2Real(w, E, mu, beta), integralI2Imag(w, E, mu, beta));
}

/*** T matrix ***/

double invTmatrixMB_real_constrained(double E, void * params) {
  double * params_arr = (double *)params;
  double w, mu, beta, a, r;
  mu = params_arr[0];
  beta = params_arr[1];
  a = params_arr[2];
  w = 0.5 * E - 2 * mu;

  r = a + integralI2Real(w, E, mu, beta);
  return r;
}

double invTmatrixMB_real(double w, void * params) {
  double * params_arr = (double *)params;
  double E, mu, beta, a, r;
  E = params_arr[0];
  mu = params_arr[1];
  beta = params_arr[2];
  a = params_arr[3];

  double y2 = - 0.25 * E + mu + 0.5 * w;

  if (y2 >= 0) {
    r = a + integralI2Real(w, E, mu, beta);
  } else {
    r = a + I1(-y2) + integralI2Real(w, E, mu, beta);
  }

  //printf("invTmatrixMB_real: %.10f, %.10f, %.10f, %.10f, %.10f\n", y2, w, E, a, r);

  return r;
}

double invTmatrixMB_imag(double w, void * params) {
  double * params_arr = (double *)params;
  double E, mu, beta, r;
  E = params_arr[0];
  mu = params_arr[1];
  beta = params_arr[2];

  double y2 = - 0.25 * E + mu + 0.5 * w;

  if (y2 > 0) {
    r = I1(y2) + integralI2Imag(w, E, mu, beta);
  } else {
    r = 0;
  }

  return r;
}

std::complex<double> invTmatrixMB(double w, double E, double mu, double beta, double a) {
  double params_arr[] = {E, mu, beta, a};
  return std::complex<double>(invTmatrixMB_real(w, params_arr), invTmatrixMB_imag(w, params_arr));
}

double polePos(double E, double mu, double beta, double a) {
  double w_hi = 0.5 * E - 2 * mu, w_lo = 0, r = 0;
  double params_arr[] = {E, mu, beta, a};
  double val1 = invTmatrixMB_real(w_hi, params_arr);
  double w_max = std::max(1e12, 1e4 * abs(a));
  bool found = false;

  if (val1 < 0) {
    return NAN;
  }

  //printf("val1: %.10f, %.10f, %.10f, %.10f\n", E, a, w_hi, val1);

  // Find a proper bound using exponential sweep.
  if (w_hi < 0) {
    for(w_lo = fmin(-1, w_hi); w_lo > -w_max; w_lo *= 2) {
      //printf("sweep: %.10f, %.10f, %.10f, %.10f, %.10f\n", E, a, w_lo, w_hi, invTmatrixMB_real(w_lo, params_arr));
      if (invTmatrixMB_real(w_lo, params_arr) < 0) {
        found = true;
        break;
      }

      w_hi = w_lo;
    }
  } else {
    double w_hi_initial = fmax(0.1 * w_hi, 1);
    for(w_lo = 0.9 * w_hi; w_lo > -w_max; w_lo = w_lo < -1 ? 2 * w_lo : w_lo - w_hi_initial) {
      //printf("sweep: %.10f, %.10f, %.10f, %.10f, %.10f\n", E, a, w_lo, w_hi, invTmatrixMB_real(w_lo, params_arr));
      if (invTmatrixMB_real(w_lo, params_arr) < 0) {
        found = true;
        break;
      }

      w_hi = w_lo;
    }
  }

  //printf("%.10f, %.10f, %.10f, %.10f\n", E, a, w_lo, w_hi);
  //printf("%.10f, %.10f, %.10f, %.10f\n", E, a, invTmatrixMB_real(w_hi, params_arr), w_hi);

  if (found) {
    int status = GSL_CONTINUE;

    gsl_function T_mat;
    T_mat.function = &invTmatrixMB_real;
    T_mat.params = params_arr;

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &T_mat, w_lo, w_hi);

    for (; status == GSL_CONTINUE; ) {
      status = gsl_root_fsolver_iterate(s);
      r = gsl_root_fsolver_root(s);
      w_lo = gsl_root_fsolver_x_lower(s);
      w_hi = gsl_root_fsolver_x_upper(s);
      status = gsl_root_test_interval(w_lo, w_hi, 0, 1e-10);
      //printf("iter: %.10f, %.10f, %.10f, %.10f, %d\n", E, a, w_lo, w_hi, status);
    }

    gsl_root_fsolver_free (s);

  } else {
    r = NAN;
  }

  return r;
}

double findLastPos(double mu, double beta, double a) {
  // Returns the maximum energy at which is possible to
  // find a solution to polePos.
  double params_arr[] = {mu, beta, a};

  if (invTmatrixMB_real_constrained(0, params_arr) <= 0) {
    return NAN;
  }

  double w_lo = 0, w_hi, r = 0;
  double w_max = 1e10;
  bool found = false;

  // Find a proper bound using exponential sweep.
  for(w_hi = 1; w_hi < w_max; w_hi *= 2) {
    if (invTmatrixMB_real_constrained(w_hi, params_arr) < 0) {
      found = true;
      break;
    }

    w_lo = w_hi;
  }

  if (found) {
    int status = GSL_CONTINUE;

    gsl_function T_mat;
    T_mat.function = &invTmatrixMB_real_constrained;
    T_mat.params = params_arr;

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &T_mat, w_lo, w_hi);

    for (; status == GSL_CONTINUE; ) {
      status = gsl_root_fsolver_iterate(s);
      r = gsl_root_fsolver_root(s);
      w_lo = gsl_root_fsolver_x_lower(s);
      w_hi = gsl_root_fsolver_x_upper(s);
      status = gsl_root_test_interval(w_lo, w_hi, 0, 1e-10);
    }

    gsl_root_fsolver_free (s);

  } else {
    r = NAN;
  }

  return r;
}

double integrandPoleRes(double x, void * params) {
  double * params_arr = (double *)params;
  double dz = params_arr[0];

  double r = integrandI2part1(x, params) / pow(x + 0.5 * dz, 2);
  return r;
}

double integralPoleRes(double E, double mu, double beta, double z0) {
  double dz = 0.5 * E - 2 * mu - z0;
  double result[] = {0}, error[] = {0};
  double params_arr[] = { dz, E, mu, beta };

  gsl_function integrand;
  integrand.function = &integrandPoleRes;
  integrand.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
  gsl_integration_qagiu(&integrand, 0, 0, 1e-10, w_size, ws, result, error);
  gsl_integration_workspace_free(ws);

  return result[0];
}

double poleRes(double E, double mu, double beta, double a) {
  double z0 = polePos(E, mu, beta, a);

  if (isnan(z0)) { return 0; }

  //assert(z1 >= z0 && "z1 has to be larger or equal than z0");

  double r, z2 = 0.25 * E - mu - 0.5 * z0;

  //printf("%.10f, %.10f, %.10f, %.10f\n", E, 1 / sqrt(2 * ( z1 - z0 )), integralPoleRes(E, mu, beta, z0), (I1dmu(z2)) / (exp(beta * z0) + 1));
  //printf("%.10f, %.10f, %.10f, %.10f, %.10f\n", E, mu, beta, a, z0);

  r = - (I1dmu(z2)) / (exp(beta * z0) + 1) / ( 0.25 / I1(z2) - integralPoleRes(E, mu, beta, z0) / M_PI );

  return r;
}

double integrandBranch(double y, void * params) {
  double * params_d = (double *)params;
  double E, mu, beta, a, final_result;
  E = params_d[0];
  mu = params_d[1];
  beta = params_d[2];
  a = params_d[3];

  double y2 = - 0.25 * E + mu + 0.5 * y;

  if (y2 < 0) {
    return 0;
  } else if (y2 == 0) {
    return std::numeric_limits<double>::infinity();
  }

  std::complex<double> r = std::complex<double>(0, -I1dmu(y2)) / invTmatrixMB(y, E, mu, beta, a);

  if constexpr (use_mpfr) {
    final_result = r.imag();
    mpfr_t mpfr_res;

    // divide final_result by (exp(beta * y) - 1)
    mpfr_init_set_d(mpfr_res, beta * y, MPFR_RNDN);
    mpfr_exp(mpfr_res, mpfr_res, MPFR_RNDN);
    mpfr_sub_ui(mpfr_res, mpfr_res, 1, MPFR_RNDN);
    mpfr_d_div(mpfr_res, final_result, mpfr_res, MPFR_RNDN);

    final_result = mpfr_get_d(mpfr_res, MPFR_RNDN);
    mpfr_clear(mpfr_res);
  } else {
    final_result = r.imag() / (exp(beta * y) - 1);
  }

  return final_result / ( 2 * M_PI );
}

double integralBranch(double E, double mu, double beta, double a) {
  double z1 = 0.5 * E - 2 * mu;
  double result[] = {0, 0}, error[] = {0, 0};
  double params_arr[] = { E, mu, beta, a };
  double eps = 1e-6;

  gsl_function integrand;
  integrand.function = &integrandBranch;
  integrand.params = params_arr;

  //auto start = std::chrono::high_resolution_clock::now();

  if (z1 < 0) {
    const size_t local_w_size = (1<<4)+(1<<1);
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_w_size);

    double x_max = - z1;
    double pts[] = { z1, 0, x_max };
    size_t n_pts = sizeof(pts) / sizeof(double);

    gsl_integration_qagp(&integrand, pts, n_pts, 0, eps, local_w_size, ws, result + 1, error + 1);
    //gsl_integration_qags(&integrand, z1, 0, 0, 1e-6, local_w_size, ws, result + 1, error + 1);
    gsl_integration_qagiu(&integrand, x_max, 0, fmax(eps, error[1]), local_w_size, ws, result, error);

    gsl_integration_workspace_free(ws);
  } else if (z1 == 0) {
    result[1] = -std::numeric_limits<double>::infinity();
  } else {
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);

    double x_max = 2 * z1;
    gsl_integration_qags(&integrand, z1, x_max, 0, eps, w_size, ws, result + 1, error + 1);
    gsl_integration_qagiu(&integrand, x_max, 0, fmax(eps, error[1]), w_size, ws, result, error);

    gsl_integration_workspace_free(ws);
  }

  //auto end = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double> dt = end-start;
  //printf("integralBranch: %.10f, %.10f, %.10f, %.1e, %.1e, %.1f s\n", z1, result[0], result[1], error[0], error[1], dt.count());

  return result[0] + result[1];
}

double integrandDensityPole(double x, void * params) {
  double * params_d = (double *)params;
  double mu, beta, a, r;
  mu = params_d[0];
  beta = params_d[1];
  a = params_d[2];

  r = poleRes(x, mu, beta, a);

  //printf("%.10f, %.10f, %.10f, %.10f, %.10f\n", x, r, mu, beta, a);

  return isnan(r) ? 0.0 : r * sqrt(x);
}

double integralDensityPole(double mu, double beta, double a) {
  double result[] = {0, 0}, error[] = {0, 0};
  double params_arr[] = { mu, beta, a };

  gsl_function integrand;
  integrand.function = &integrandDensityPole;
  integrand.params = params_arr;

  auto start = std::chrono::high_resolution_clock::now();

  if (a <= 0) {
    double E_max = findLastPos(mu, beta, a);
    //printf("limit: %.10f, %.10f\n", a, E_max);

    if (!isnan(E_max)) {
      gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
      gsl_integration_qag(&integrand, 0, E_max, 0, 1e-10, w_size, GSL_INTEG_GAUSS31, ws, result, error);
      gsl_integration_workspace_free(ws);
    }
  } else {
    double E_max = 16 / beta + 4 * (mu + a*a);

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
    gsl_integration_qag(&integrand, 0, E_max, 0, 1e-10, w_size, GSL_INTEG_GAUSS51, ws, result, error);
    gsl_integration_qagiu(&integrand, E_max, 0, 1e-10, w_size, ws, result + 1, error+ 1);
    gsl_integration_workspace_free(ws);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = end-start;
  printf("integralDensityPole: %.10f, %.10f, %.10f, %.3e, %.3e, %.3f s\n", a, result[0], result[1], error[0], error[1], dt.count());

  return (result[0] + result[1]) / (4 * M_PI * M_PI);
}

double integrandDensityBranch(double x, void * params) {
  double mu, beta, a, r;
  double * params_d = (double *)params;
  mu = params_d[0];
  beta = params_d[1];
  a = params_d[2];

  r = sqrt(x) * integralBranch(x, mu, beta, a);

  return r;
}

double integralDensityBranch(double mu, double beta, double a) {
  double result[] = {0, 0}, error[] = {0, 0};
  double params_arr[] = { mu, beta, a };

  gsl_function integrand;
  integrand.function = &integrandDensityBranch;
  integrand.params = params_arr;

  auto start = std::chrono::high_resolution_clock::now();

  /*
  size_t neval = 0;
  gsl_integration_qng(&integrand, 0, x_max, 0, 1e-10, result, error, &neval);
  */

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);

  if (mu > 0) {
    double x_max = 8 * mu;
    double pts[] = { 0, 4 * mu, x_max };
    size_t n_pts = sizeof(pts) / sizeof(double);

    gsl_integration_qagp(&integrand, pts, n_pts, 0, 1e-6, w_size, ws, result, error);
    gsl_integration_qagiu(&integrand, x_max, 0, error[0], w_size, ws, result + 1, error + 1);
  } else {
    //size_t neval;
    //double x_final = 40;
    //gsl_integration_qng(&integrand, 0, x_final, 0, 1e-6, result, error, &neval);
    gsl_integration_qagiu(&integrand, 0, 0, 1e-6, w_size, ws, result + 1, error + 1);
  }

  gsl_integration_workspace_free(ws);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = end-start;
  printf("integralDensityBranch: %.10f, %.10f, %.10f, %.1e, %.1e, %.1f s\n", a, result[0], result[1], error[0], error[1], dt.count());

  return result[0] + result[1];
}

double analytic_n_id(double mu_i, double m_ratio) {
  // m_ratio = m_r / m_i
  return 2 * std::pow(m_ratio, -1.5) * polylogExpM(1.5, mu_i / (4 * M_PI));
}

double analytic_n_ex(double mu, double m_ratio, double a) {
  // m_ratio = m_e / m_r + m_e / m_r
  // mu = mu_e + mu_h
  if (a < 0) { return 0; }
  return 4 * std::pow(m_ratio, 1.5) * polylogExp(1.5, (mu + a*a / 4) / (4 * M_PI));
}

double analytic_n_sc_f(double y, void * params) {
  // mu = mu_e + mu_h
  double * params_arr = (double*)params;
  double mu = params_arr[0];
  double a = params_arr[1];

  return polylogExp(1.5, mu / (4 * M_PI) - y*y) / (y*y + a*a / (16 * M_PI));
}

double analytic_n_sc(double mu, double m_ratio, double a) {
  // m_ratio = m_e / m_r + m_e / m_r
  // mu = mu_e + mu_h
  constexpr uint32_t local_ws_size = 1<<8;
  double result[2] = {0}, err[2] = {0};
  double params_arr[] = {mu, a};

  double x_max = 0;

  if (a < 1e-4) {
    x_max = 1e-5;

    gsl_function integrand;
    integrand.function = &analytic_n_sc_f;
    integrand.params = params_arr;

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
    gsl_integration_qags(&integrand, 0, x_max, 0, 1e-10, w_size, ws, result, err);
    gsl_integration_workspace_free(ws);
  }

  result[1] = integralTemplate<local_ws_size>(&analytic_n_sc_f, x_max, mu, a);
  return - a * std::pow(m_ratio / M_PI, 1.5) * (result[0] + result[1]);
}

int analytic_mu_param_f(const gsl_vector * x, void * params, gsl_vector * f) {
  /*
    Here we need to solve two equations,

    |  n = n_id + n_ex + n_sc
    |  n_e,id = n_h,id

    solving for mu_e and mu_h in the
    process, for a fixed value of a.
  */
  constexpr uint32_t n_eq = 2;

  double * params_arr = (double*)params;
  double n_dless = params_arr[0];
  double m_pe = params_arr[1];
  double m_ph = params_arr[2];
  double a = params_arr[3];

  double mu_e = gsl_vector_get(x, 0);
  double mu_h = gsl_vector_get(x, 1);

  double mu = mu_e + mu_h, m_ratio = 1/m_pe + 1/m_ph;

  double yv[n_eq] = {0};

  double n_id[] = {analytic_n_id(mu_e, m_pe), analytic_n_id(mu_h, m_ph)};

  yv[0] = - n_dless + n_id[0] + n_id[1] + analytic_n_ex(mu, m_ratio, a) + analytic_n_sc(mu, m_ratio, a);
  yv[1] = n_id[0] - n_id[1];

  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }

  return GSL_SUCCESS;
}

std::vector<double> analytic_mu_param(double n_dless, double m_pe, double m_ph, double a) {
  // n = number of equations
  constexpr size_t n = 2;
  double params_arr[] = {n_dless, m_pe, m_ph, a};
  gsl_multiroot_function f = {&analytic_mu_param_f, n, params_arr};

  std::vector<double> mu_i = ideal_c_mu(n_dless, m_pe, m_ph);

  double * x_init = mu_i.data();
  gsl_vector * x = gsl_vector_alloc(n);

  for(size_t i = 0; i < n; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fsolver_type * T = gsl_multiroot_fsolver_hybrids;
  gsl_multiroot_fsolver * s = gsl_multiroot_fsolver_alloc(T, n);
  gsl_multiroot_fsolver_set(s, &f, x);

  for (int status = GSL_CONTINUE; status == GSL_CONTINUE;) {
    status = gsl_multiroot_fsolver_iterate(s);
    if (status) { break; }
    status = gsl_multiroot_test_residual(s->f, 1e-10);
  }

  std::vector<double> r(n);

  for(size_t i = 0; i < n; i++) { r[i] = gsl_vector_get(s->x, i); }

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return r;
}

double analytic_mu_param_b_f(double mu_e, void * params) {
  /*
    Here we need to solve one equation,

    |  n = n_id + n_ex + n_sc

    solving for mu_e (and mu_h) in the
    process, for a fixed value of a and n.
  */
  double * params_arr = (double*)params;
  double n = params_arr[0];
  double m_pe = params_arr[1];
  double m_ph = params_arr[2];
  double a = params_arr[3];
  double m_sigma = params_arr[4];

  double mu_h{ideal_mu_b(mu_e, m_ph, m_pe)};
  double mu_t = mu_e + mu_h;

  double n_id{analytic_n_id(mu_e, m_pe)};
  //printf("mu: %.5f, %.5f, %.10f\n", mu_e, mu_h, analytic_n_ex(mu_t, m_sigma, a));

  return - n + n_id + analytic_n_ex(mu_t, m_sigma, a) + analytic_n_sc(mu_t, m_sigma, a);
}

std::vector<double> analytic_mu_param_b(double n, double m_pe, double m_ph, double a) {
  // n = number of equations
  double m_sigma = 1/m_pe + 1/m_ph;
  double params_arr[] = {n, m_pe, m_ph, a, m_sigma};
  double z, z_max, z_min;

  gsl_function funct;
  funct.function = &analytic_mu_param_b_f;
  funct.params = params_arr;

  // TODO: maybe take z_min and z_max as arguments?
  if (a >= 0) {
    // This bound is only valid when a > 0.
    z_max = ideal_mu_v(-0.25 * a*a - global_eps, -1, m_pe, m_ph);
    z_min = - 0.25 * a * a + 4 * M_PI * invPolylogExp(1.5, 0.25 * std::pow(m_sigma, -1.5) * n);
  } else {
    z_max = ideal_mu_v(0, -1, m_pe, m_ph);
    z_min = 4 * M_PI * invPolylogExp(1.5, 0.25 * std::pow(m_sigma, -1.5) * n);
  }

  /*
  double f0{funct.function(z_min, params_arr)}, f1{funct.function(z_max, params_arr)};
  if (f0 * f1 > 0) {
    z = std::min(std::abs(f0), std::abs(f1));
    return std::vector<double>({z, ideal_mu_b(z, m_ph, m_pe)});
  }
  */

  //printf("first: %.3f, %.10f, %.10f, %.2f, %.2f\n", a, z_min, z_max, funct.function(z_min, params_arr), funct.function(z_max, params_arr));

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);

    //printf("iter %d: %.3f, %.10f, %.10f, %.7e\n", iter, a, z_min, z_max, funct.function(z, params_arr));
  }

  //printf("result: %.2f, %.10f, %.10f, %.10f\n", a, z, fluct_Ec_a_f(z, params_arr), fluct_Ec_a_df(z, params_arr));

  gsl_root_fsolver_free(s);
  return std::vector<double>({z, ideal_mu_b(z, m_ph, m_pe)});
}

std::vector<double> analytic_mu_param_dn(double n_dless, double m_pe, double m_ph, double a) {
  return derivative_c2<2>(&analytic_mu_param, n_dless, n_dless * 1e-6, m_pe, m_ph, a);
}

int analytic_mu_f(const gsl_vector * x, void * params, gsl_vector * f) {
  /*
    Here we need to solve two equations,

    | n = n_id + n_ex + n_sc
    | a(n_dless, mu_param) == a_param

    solving for a and mu_i in the
    process.
  */
  constexpr uint32_t n_eq = 3;

  double * params_arr = (double*)params;
  double n_dless = params_arr[0];
  double m_pe = params_arr[1];
  double m_ph = params_arr[2];
  double eps_r = params_arr[3];
  double e_ratio = params_arr[4];

  double mu_e = gsl_vector_get(x, 0);
  double mu_h = gsl_vector_get(x, 1);
  double a = gsl_vector_get(x, 2);

  double yv[n_eq] = {0};
  std::vector<double> mu_i = analytic_mu_param(n_dless, m_pe, m_ph, a);
  std::vector<double> lambda_i = analytic_mu_param_dn(n_dless, m_pe, m_ph, a);
  double lambda_s = 1 / std::sqrt(std::abs(1 / lambda_i[0] + 1 / lambda_i[1]));
  //printf("%.10f, %.10f,  %.10f\n", n_dless, lambda_i[0], lambda_i[1]);

  yv[0] = mu_i[0] - mu_e;
  yv[1] = mu_i[1] - mu_h;
  yv[2] = 1/wavefunction_int(eps_r, e_ratio, lambda_s) - a;

  for (uint32_t i = 0; i < n_eq; i++) { gsl_vector_set(f, i, yv[i]); }

  return GSL_SUCCESS;
}

std::vector<double> analytic_mu(double n_dless, double m_pe, double m_ph, double eps_r, double e_ratio) {
  // n = number of equations
  // TODO: check the prefactor of the chemical potential.
  constexpr size_t n = 3;
  double params_arr[] = {n_dless, m_pe, m_ph, eps_r, e_ratio};
  gsl_multiroot_function f = {&analytic_mu_f, n, params_arr};

  std::vector<double> mu_id = ideal_c_mu(n_dless, m_pe, m_ph);
  double a = 0.5 / std::sqrt(mu_id[0] + mu_id[1]);
  std::vector<double> mu_i = analytic_mu_param(n_dless, m_pe, m_ph, a);
  double x_init[] = { mu_i[0], mu_i[1], a };

  gsl_vector * x = gsl_vector_alloc(n);

  for(size_t i = 0; i < n; i++) { gsl_vector_set(x, i, x_init[i]); }

  const gsl_multiroot_fsolver_type * T = gsl_multiroot_fsolver_hybrids;
  gsl_multiroot_fsolver * s = gsl_multiroot_fsolver_alloc(T, n);
  gsl_multiroot_fsolver_set(s, &f, x);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE &&  iter < 32; iter++) {
    status = gsl_multiroot_fsolver_iterate(s);
    if (status) { break; }
    status = gsl_multiroot_test_residual(s->f, 1e-10);
  }

  std::vector<double> r(n);

  for(size_t i = 0; i < n; i++) { r[i] = gsl_vector_get(s->x, i); }

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return r;
}

double yukawa_pot(double x, double eps_r, double e_ratio, double lambda_s) {
  /* e_ratio = m_r * c^2 / e_th */
  // alpha = fine-structure constant ~ 1 / 137
  const double alpha{7.2973525664e-3};
  return - alpha / eps_r * std::sqrt(2 * e_ratio) * std::exp(- 4 * std::sqrt(alpha * M_PI / eps_r * std::sqrt(e_ratio / 8)) * x / lambda_s) / x;
}

wavefunction_ode::wavefunction_ode(double eps_r, double e_ratio, double lambda_s) : eps_r(eps_r), e_ratio(e_ratio), lambda_s(lambda_s) {}

void wavefunction_ode::operator()(const std::array<double, 2> &y, std::array<double, 2> &dy, double x) {
  dy[0] = y[1];
  dy[1] = yukawa_pot(x, eps_r, e_ratio, lambda_s) * y[0];
}

double wavefunction_int(double eps_r, double e_ratio, double lambda_s) {
  std::array<double, 2> y{{0.0, 1.0}};
  double x0{1e-10}, x1{1<<8};
  double a0{x1}, a1;
  const double err = 1e-10;

  typedef boost::numeric::odeint::runge_kutta_cash_karp54< std::array<double, 2> > error_stepper_type;
  typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

  controlled_stepper_type controlled_stepper;
  wavefunction_ode wf(eps_r, e_ratio, lambda_s);

  for(uint8_t i = 0, status = 0; status == 0 && i < 32; i++) {
    integrate_adaptive(controlled_stepper, wf, y, x0, x1, x1 * 1e-3);

    a1 = x1 - y[0] / y[1];

    //printf("%.2f, %.10f, %.10f, %.2e\n", lambda_s, a1, a0, a1 - a0);

    if (2 * std::abs(a1 - a0) / (a0 + a1) > err) {
      x0 = x1;
      x1 *= 2;

      a0 = a1;
    } else {
      break;
    }
  }

  return a1;
}

double ideal_mu(double n, double m_pa) {
  // m_ratio = m_r / m_i
  return 4 * M_PI * invPolylogExpM(1.5, 0.25 * n * std::pow(m_pa, 1.5));
}

std::vector<double> ideal_c_mu(double n, double m_pe, double m_ph) {
  return std::vector<double>({ideal_mu(n, m_pe), ideal_mu(n, m_ph)});
}

double ideal_mu_dn_f(double z, void * params) {
  double * params_arr = (double*)(params);
  double s = params_arr[0];
  double m_ratio = params_arr[1];

  return invPolylogExpM(s, 0.25 * z * std::pow(m_ratio, 1.5));
}

double ideal_mu_dn(double n_dless, double m_ratio) {
  // m_ratio = m_p / m_i

  gsl_function F;
  double r, err;
  double params_arr[] = {1.5, m_ratio};

  F.function = &ideal_mu_dn_f;
  F.params = params_arr;

  gsl_deriv_forward(&F, n_dless, n_dless * 1e-6, &r, &err);

  //printf("%.3e, %.3e, %.3e\n", n_dless, 4 * M_PI * arg * r, err);

  return 4 * M_PI * r;
}

double ideal_mu_b(double mu_a, double m_pb, double m_pa) {
  return 4 * M_PI * invPolylogExpM(1.5, std::pow(m_pb / m_pa, 1.5) * polylogExpM(1.5, mu_a / (4 * M_PI)));
}

double ideal_mu_v_f(double mu_e, void * params) {
  double * params_arr = (double*)params;
  double v = params_arr[0];
  double m_pe = params_arr[1];
  double m_ph = params_arr[2];

  return mu_e + ideal_mu_b(mu_e, m_ph, m_pe) - v;
}

double ideal_mu_v_df(double mu_e, void * params) {
  double * params_arr = (double*)params;
  double m_pe = params_arr[1];
  double m_ph = params_arr[2];

  return 1 + derivative_c2<1>(&ideal_mu_b, mu_e, global_eps * mu_e, m_ph, m_pe)[0];
}

void ideal_mu_v_fdf(double z, void * params, double * f, double * df) {
  f[0] = ideal_mu_v_f(z, params);
  df[0] = ideal_mu_v_df(z, params);
}

double ideal_mu_v(double v, double mu_0, double m_pe, double m_ph) {
  /*
    Solves the equation mu_e + mu_h == v for mu_e,
    assuming n_id,e == n_id,h.
  */

  double params_arr[] = {v, m_pe, m_ph};
  double z0, z{mu_0};

  gsl_function_fdf funct;
  funct.f = &ideal_mu_v_f;
  funct.df = &ideal_mu_v_df;
  funct.fdf = &ideal_mu_v_fdf;
  funct.params = params_arr;

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

