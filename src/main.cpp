#include "common.h"

const mp_prec_t prec = 64;
const size_t w_size = 1<<10;

double I1(double z2) {
  return - sqrt(z2);
}

double I1dmu(double z2) {
  return 1 / sqrt(z2);
}

double logExp(double x, double xmax = 50) {
  // log( 1 + exp( x ) )
  double d_y;

  if (x < xmax) {
    // Approximate log(exp(x) + 1) ~ x when x > xmax
    /*
    mpfr_t y;
    mpfr_init_set_d(y, x, MPFR_RNDN);
    mpfr_exp(y, y, MPFR_RNDN);
    mpfr_add_ui(y, y, 1, MPFR_RNDN);
    mpfr_log(y, y, MPFR_RNDN);

    d_y = mpfr_get_d(y, MPFR_RNDN);

    mpfr_clear(y);
    */

    d_y = log(1+exp(x));
  } else {
    d_y = x;
  }

  return d_y;
}

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

double integrandBranch(double y, void * params) {
  double * params_d = (double *)params;
  double E, mu, beta, a;
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

  /*
  double final_result = r.imag() / (2 * M_PI);
  mpfr_t mpfr_res;
  // divide final_result by (exp(beta * y) - 1)
  mpfr_init_set_d(mpfr_res, beta * y, MPFR_RNDN);
  mpfr_exp(mpfr_res, mpfr_res, MPFR_RNDN);
  mpfr_sub_ui(mpfr_res, mpfr_res, 1, MPFR_RNDN);
  mpfr_d_div(mpfr_res, final_result, mpfr_res, MPFR_RNDN);

  final_result = mpfr_get_d(mpfr_res, MPFR_RNDN);
  mpfr_clear(mpfr_res);
  //printf("%.16f, %.16f\n", y, final_result);
  */

  double final_result = r.imag() / (2 * M_PI * (exp(beta * y) - 1));

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

int main(/*int argc, char ** argv*/)
{
  gsl_set_error_handler_off();
  mpfr_set_default_prec(prec);

  double E = 1, mu = -1, beta = 2, a = 1;

  uint32_t N = 1<<10;

  auto start = std::chrono::high_resolution_clock::now();

  /*
  std::complex<double> r;
  for (uint32_t i = 0; i < N; i++) {
    r = integralBranch(E, mu, beta, a);
  }
  */

  double r;
  for (uint32_t i = 0; i < N; i++) {
    r = polePos(E, mu, beta, a);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = end-start;

  printf("Pole position: %.10f\n", r);
  printf("(%d) %0.3f μs\n", N, dt.count() / N * 1e6);

  return 0;
}
