#include "integrals.h"

int main(/*int argc, char ** argv*/)
{
  initializeMPFR_GSL();

  double p1, p2;
  p1 = polylogExp(1.5, 1);
  p2 = polylogExpM(1.5, 1);

  printf("%.16f, %.16f", p1, p2);

  std::cout << "\nFinished\n";

  return 0;

  double w = 3, E = 0, mu = -1, beta = 2, a = 1;

  uint32_t N = 1<<10;

  //double z0 = polePos(E, mu, beta, a);

  auto start = std::chrono::high_resolution_clock::now();

  /*
  double r;
  for (uint32_t i = 0; i < N; i++) {
    r = polePos(E, mu, beta, a);
  }

  double r;
  for (uint32_t i = 0; i < N; i++) {
    r = poleRes_pole(E, mu, beta, a, z0);
  }

  std::complex<double> r;
  double params[] = {E, mu, beta, a};
  for (uint32_t i = 0; i < N; i++) {
    r = invTmatrixMB_real(w, params);
  }

  double r;
  for (uint32_t i = 0; i < N; i++) {
    r = integralDensityPole(mu, beta, a);
  }
  */

  std::complex<double> r;
  for (uint32_t i = 0; i < N; i++) {
    r = integralBranch(E, mu, beta, a);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = end-start;

  std::complex<double> r2(r);
  printf("result: (%.10f, %.10f)\n", r2.real(), r2.imag());
  printf("(%d) %0.3f μs\n", N, dt.count() / N * 1e6);

  return 0;
}
