#include "common.h"

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
  double w, E, mu, beta, r;
  w = params_arr[0];
  E = params_arr[1];
  mu = params_arr[2];
  beta = params_arr[3];

  if (E < 1e-6) {
    r = sqrt(x) / (exp(beta * (x - mu)) + 1);
  } else {
    double E_ex = 0.25 * E - mu + x;
    double Ep = beta * (E_ex + sqrt( E * x )), Em = beta * (E_ex - sqrt( E * x ));

    r = sqrt(x) + (logExp(Em) - logExp(Ep)) / ( 2 * beta * sqrt(E) );
    if (r != r) { return 0; }
    r = abs(r) < 1e-15 ? 0.0 : r;
  }

  return r;
}

double integrandI2part2(double x, void * params) {
  double * params_arr = (double *)params;
  double w, E, mu, beta, z2;
  w = params_arr[0];
  E = params_arr[1];
  mu = params_arr[2];
  beta = params_arr[3];
  z2 = params_arr[4];

  return integrandI2part1(x, params) / (x - z2);
}

double integralI2Real(double w, double E, double mu, double beta) {
  double result[2] = {0, 0}, error;
  double params_arr[] = {w, E, mu, beta, 0.5 * w - 0.25 * E + mu};

  gsl_function integrand_part2;
  integrand_part2.function = &integrandI2part2;
  integrand_part2.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);

  if (params_arr[4] < 0) {
    gsl_integration_qagiu(&integrand_part2, 0, 0, 1e-10, w_size, ws, result, &error);
  } else {
    gsl_function integrand_part1;
    integrand_part1.function = &integrandI2part1;
    integrand_part1.params = params_arr;

    gsl_integration_qawc(&integrand_part1, 0, 2 * params_arr[4], params_arr[4], 0, 1e-10, w_size, ws, result, &error);
    gsl_integration_qagiu(&integrand_part2, 2 * params_arr[4], 0, 1e-10, w_size, ws, result + 1, &error);
  }

  gsl_integration_workspace_free(ws);

  return 2 / M_PI * (result[0] + result[1]);
}

double integralI2Imag(double w, double E, double mu, double beta) {
  double params_arr[] = {w, E, mu, beta, 0.5 * w - 0.25 * E + mu};

  if (params_arr[4] < 0) { return 0; }

  return 2 * integrandI2part1(params_arr[4], params_arr);
}

std::complex<double> I2(double w, double E, double mu, double beta) {
  return std::complex<double>(integralI2Real(w, E, mu, beta), integralI2Imag(w, E, mu, beta));
}

/*** T matrix ***/

std::complex<double> invTmatrixMB(double w, double E, double mu, double beta, double a) {
  double y2 = - 0.25 * E + mu + 0.5 * w;
  std::complex<double> r;

  if (y2 > 0) {
    r = std::complex<double>(a, I1(y2)) + I2(w, E, mu, beta);
  } else {
    r = a + I1(-y2) + I2(w, E, mu, beta);
  }

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

  if (y2 > 0) {
    r = a + integralI2Real(w, E, mu, beta);
  } else {
    r = a + I1(-y2) + integralI2Real(w, E, mu, beta);
  }

  return r;
}

double polePos(double E, double mu, double beta, double a) {
  double w_lo = 0.5 * E - 2 * mu, w_hi, r = 0;
  std::complex<double> val1 = invTmatrixMB(w_lo, E, mu, beta, a);
  bool found = false;

  assert(abs(val1.imag()) < 1e-10 && "Imaginary part of T_MB is not zero.");

  if (invTmatrixMB(-1e10, E, mu, beta, a).real() * val1.real() > 0) {
    return NAN;
  }

  // Find a proper bound using exponential sweep.
  for(w_hi = - 1; w_hi > -1e10; w_hi *= 2) {
    if (invTmatrixMB(w_hi, E, mu, beta, a).real() * val1.real() < 0) {
      found = true;
      break;
    }
    w_lo = w_hi;
  }

  if (found) {
    double params[] = {E, mu, beta, a};
    int status = GSL_CONTINUE;

    gsl_function T_mat;
    T_mat.function = &invTmatrixMB_real;
    T_mat.params = params;

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &T_mat, w_hi, w_lo);

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
  double E, mu, beta, z0, r;
  z0 = params_arr[0];
  E = params_arr[1];
  mu = params_arr[2];
  beta = params_arr[3];

  r = integrandI2part1(x, params) / pow(x + 0.5 * (0.5 * E - 2 * mu - z0), 2);

  return r;
}

double integralPoleRes(double E, double mu, double beta, double z0) {
  double result[] = {0, 0}, error;
  double params_arr[] = {z0, E, mu, beta};

  gsl_function integrand;
  integrand.function = &integrandPoleRes;
  integrand.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);

  gsl_integration_qags(&integrand, 0, 2 * z0, 0, 1e-10, w_size, ws, result, &error);
  gsl_integration_qagiu(&integrand, 2 * z0, 0, 1e-10, w_size, ws, result + 1, &error);

  gsl_integration_workspace_free(ws);

  return result[0] + result[1];
}

double poleRes(double E, double mu, double beta, double a) {
  double z1 = 0.5 * E - 2 * mu;
  double z0 = polePos(E, mu, beta, a);

  if (isnan(z0)) { return 0; }

  assert(z1 > z0 && "z1 has to be larger than z0");

  double r, z2 = 0.5 * ( z1 - z0 );

  r = (I1dmu(z2)) / (exp(beta * z0) + 1) / ( 1 / sqrt(2 * ( z1 - z0 )) - integralPoleRes(E, mu, beta, z0) / M_PI );

  return r;
}

double poleRes(double E, double mu, double beta, double /*a*/, double z0) {
  double z1 = 0.5 * E - 2 * mu;

  assert(z1 > z0 && "z1 has to be larger than z0");

  double r, z2 = 0.5 * ( z1 - z0 );

  r = (I1dmu(z2)) / (exp(beta * z0) + 1) / ( 1 / sqrt(2 * ( z1 - z0 )) - integralPoleRes(E, mu, beta, z0) / M_PI );

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
    printf("%.16f\n", y);

    return 0;
  }

  std::complex<double> r = std::complex<double>(0, -I1dmu(y2)) / invTmatrixMB(y, E, mu, beta, a);

  if constexpr (use_mpfr) {
    final_result = r.imag() / (2 * M_PI);
    mpfr_t mpfr_res;

    // divide final_result by (exp(beta * y) - 1)
    mpfr_init_set_d(mpfr_res, beta * y, MPFR_RNDN);
    mpfr_exp(mpfr_res, mpfr_res, MPFR_RNDN);
    mpfr_sub_ui(mpfr_res, mpfr_res, 1, MPFR_RNDN);
    mpfr_d_div(mpfr_res, final_result, mpfr_res, MPFR_RNDN);

    final_result = mpfr_get_d(mpfr_res, MPFR_RNDN);
    mpfr_clear(mpfr_res);
  } else {
    final_result = r.imag() / (2 * M_PI * (exp(beta * y) - 1));
  }

  return final_result;
}

double integralBranch(double E, double mu, double beta, double a) {
  double result[2] = {0, 0}, error;
  double z1 = 0.5 * E - 2 * mu;
  double params_arr[] = {E, mu, beta, a};

  gsl_function integrand;
  integrand.function = &integrandBranch;
  integrand.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);

  gsl_integration_qags(&integrand, z1, 2 * z1, 0, 1e-10, w_size, ws, result, &error);
  gsl_integration_qagiu(&integrand, 2 * z1, 0, 1e-10, w_size, ws, result + 1, &error);

  gsl_integration_workspace_free(ws);

  return result[0] + result[1];
}
