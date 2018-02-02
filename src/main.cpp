#include "common.h"
#include "integrals.h"

int main(/*int argc, char ** argv*/)
{
  initializeMPFR_GSL();

  double w = 0.45, E = 1, mu = 1, beta = 1, a = -1;
  std::vector<double> rf(2);

  for (uint32_t i = 0; i < 10; i++) {
    rf = analytic_mu(w + i * 0.05, 1.4745762712, 3.1071428571, 6.56, 4.7165741283e+07);
    printf("%.10f, %.10f, %.10f, %.10f\n", w+ i * 0.15, rf[0], rf[1], rf[2]);
  }

  return 0;

  uint32_t N = 1<<3;

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

  std::complex<double> r;
  for (uint32_t i = 0; i < N; i++) {
    r = integralBranch(E, mu, beta, a);
  }
  */

  //double r{0};
  std::complex<double> r;
  for (uint32_t i = 0; i < N; i++) {
    //r = invPolylogExp(0.5, 1e-2);
    analytic_mu_param_dn(w, 0.1, 10, a);
    //r = wavefunction_int(1, 1e2, 1e3);
    //r = polylogExpM(1.5, 1);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = end-start;

  std::complex<double> r2(r);
  printf("result: (%.10f, %.10f)\n", r2.real(), r2.imag());
  printf("(%d) %0.3f μs\n", N, dt.count() / N * 1e6);

  return 0;
}
