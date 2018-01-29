#pragma once

#include "common.h"
#include "templates.h"

extern "C" {
  /*** I1 ***/

  double I1(double z2);
  double I1dmu(double z2);

  /*** I2 ***/

  double integrandI2part1(double x, void * params);
  double integrandI2part2(double x, void * params);
  double integrandI2part2v2(double x, void * params);
  double integralI2Real(double w, double E, double mu, double beta);
  double integralI2Imag(double w, double E, double mu, double beta);

  // TODO: I2dmu
  double integrandI2dmuPart1(double x, void * params);

  /*** T matrix ***/

  double invTmatrixMB_real(double w, void * params);
  double invTmatrixMB_imag(double w, void * params);

  /*** Matsubara sum: pole contribution ***/

  double polePos(double E, double mu, double beta, double a);
  double findLastPos(double mu, double beta, double a);
  double integrandPoleRes(double x, void * params);
  double integralPoleRes(double E, double mu, double beta, double z0);
  double poleRes(double E, double mu, double beta, double a);

  /*** Matsubara sum: branch contribution ***/

  double integrandBranch(double y, void * params);
  double integralBranch(double E, double mu, double beta, double a);

  /*** Density: pole contribution ***/

  double integrandDensityPole(double x, void * params);
  double integralDensityPole(double mu, double beta, double a);

  /*** Density: branch contribution ***/

  double integrandDensityBranch(double x, void * params);
  double integralDensityBranch(double mu, double beta, double a);

  /*** Density: analytic solution ***/

  double analytic_n_id(double mu, double beta);
  double analytic_n_ex(double mu, double beta, double a);
  double analytic_n_sc(double mu, double beta, double a);

  double analytic_n(double mu, void * params);
  double analytic_mu(double beta, double a);

  /*** Scattering length: ODE ***/

  double yukawa_pot(double x, double eps_r, double e_ratio, double lambda_s);
  double wavefunction_int(double eps_r, double e_ratio, double lambda_s);

  /*** Scattering length: ideal gas chemical potential ***/

  double mu_ideal(double n_dless, double m_ratio);
  double mu_ideal_dn(double n_dless, double m_ratio);
}

/*** C++ only ***/

std::complex<double> I2(double w, double E, double mu, double beta);

std::complex<double> invTmatrixMB(double w, double E, double mu, double beta, double a);

class wavefunction_ode {
private:
  double eps_r{0.0};
  double e_ratio{0.0};
  double lambda_s{0.0};

public:
  wavefunction_ode(double eps_r, double e_ratio, double lambda_s);
  void operator()(const std::array<double, 2> &y, std::array<double, 2> &dy, double x);
};

