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
  } else {
    gsl_function integrand_part1;
    integrand_part1.function = &integrandI2part1;
    integrand_part1.params = params_arr;

    double x_max = 4 * fmax(1e-4, params_arr[4]);

    gsl_integration_qawc(&integrand_part1, 0, x_max, params_arr[4], 0, 1e-10, local_w_size, ws, result, error);
    gsl_integration_qagiu(&integrand_part2, x_max, 0, error[0], local_w_size, ws, result + 1, error + 1);
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
    r = a;// + integralI2Real(w, E, mu, beta);
  } else {
    r = a + I1(-y2);// + integralI2Real(w, E, mu, beta);
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
  double z1 = 0.5 * E - 2 * mu;
  double z0 = polePos(E, mu, beta, a);

  if (isnan(z0)) { return 0; }

  //assert(z1 >= z0 && "z1 has to be larger or equal than z0");

  double r, z2 = 2 * ( z1 - z0 );

  //printf("%.10f, %.10f, %.10f, %.10f\n", E, 1 / sqrt(2 * ( z1 - z0 )), integralPoleRes(E, mu, beta, z0), (I1dmu(z2)) / (exp(beta * z0) + 1));
  //printf("%.10f, %.10f, %.10f, %.10f, %.10f\n", E, mu, beta, a, z0);

  r = (I1dmu(z2)) / (exp(beta * z0) + 1) / ( I1dmu(z2) - integralPoleRes(E, mu, beta, z0) / M_PI );

  return r;
}

double poleRes_pole(double E, double mu, double beta, double /*a*/, double z0) {
  double z1 = 0.5 * E - 2 * mu;

  //assert(z1 >= z0 && "z1 has to be larger or equal than z0");

  double r, z2 = 2 * ( z1 - z0 );

  r = (I1dmu(z2)) / (exp(beta * z0) + 1) / ( I1dmu(z2) - integralPoleRes(E, mu, beta, z0) / M_PI );

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

  //auto start = std::chrono::high_resolution_clock::now();

  if (a <= 0) {
    double E_max = findLastPos(mu, beta, a);
    //printf("limit: %.10f, %.10f\n", a, E_max);

    if (!isnan(E_max)) {
      //E_max -= 1e-10;
      /*
      gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
      gsl_integration_qag(&integrand, 0, E_max, 0, 1e-10, w_size, GSL_INTEG_GAUSS15, ws, result, &error);
      gsl_integration_workspace_free(ws);
      */

      size_t neval = 0;
      gsl_integration_qng(&integrand, 0, E_max, 0, 1e-10, result, error, &neval);
    }
  } else {
    double x_max = 16 / beta + 4 * (mu + a*a);

    ///*
    size_t neval = 0;
    gsl_integration_qng(&integrand, 0, x_max, 0, 1e-10, result, error, &neval);
    //*/

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
    //gsl_integration_qag(&integrand, 0, x_max, 0, 1e-10, w_size, GSL_INTEG_GAUSS31, ws, result, &error);
    gsl_integration_qagiu(&integrand, x_max, 0, 0.5 * error[0], w_size, ws, result + 1, error + 1);
    gsl_integration_workspace_free(ws);
  }

  //auto end = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double> dt = end-start;
  //printf("integralDensityPole: %.10f, %.10f, %.10f, %.3e, %.3e, %.3f s\n", a, result[0], result[1], error[0], error[1], dt.count());

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

double analytic_n_id(double mu, double beta) {
  return polylogExpM(1.5, beta * mu) / std::pow(4 * M_PI * beta, 1.5);
}

double analytic_n_ex(double mu, double beta, double a) {
  if (a < 0) {
    return 0;
  }

  return 4 * polylogExp(1.5, beta * (2 * a*a + 2 * mu)) / std::pow(4 * M_PI * beta, 1.5);
}

double integrandAnalytic_n_sc(double y, void * params) {
  double * arr_params = (double*)params;
  double mu = arr_params[0];
  double beta = arr_params[1];
  double a = arr_params[2];

  return polylogExp(1.5, beta * (-0.5*y*y + 2 * mu)) / (y*y + std::pow(4 / a, 2));
}

double analytic_n_sc(double mu, double beta, double a) {
  double result = integralTemplate(&integrandAnalytic_n_sc, mu, beta, a);
  return - 2 * a / std::pow(M_PI, 3) * result;
}

