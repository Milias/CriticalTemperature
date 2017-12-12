#include "integrals.h"

int main(/*int argc, char ** argv*/)
{
  gsl_set_error_handler_off();
  mpfr_set_default_prec(prec);

  double w = 3, E = 0, mu = -1, beta = 2, a = -0.01;

  uint32_t N = 1<<14;

  //double z0 = polePos(E, mu, beta, a);

  auto start = std::chrono::high_resolution_clock::now();

  /*
  std::complex<double> r;
  for (uint32_t i = 0; i < N; i++) {
    r = integralBranch(E, mu, beta, a);
  }

  double r;
  for (uint32_t i = 0; i < N; i++) {
    r = poleRes_pole(E, mu, beta, a, z0);
  }

  std::complex<double> r;
  for (uint32_t i = 0; i < N; i++) {
    r = I2(w, E, mu, beta);
  }
  */

  double r;
  for (uint32_t i = 0; i < N; i++) {
    r = polePos(E, mu, beta, a);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = end-start;

  std::complex<double> r2(r);
  printf("result: (%.10f, %.10f)\n", r2.real(), r2.imag());
  printf("(%d) %0.3f μs\n", N, dt.count() / N * 1e6);

  return 0;
}
