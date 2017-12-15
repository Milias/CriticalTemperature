#pragma once

#include "common.h"

extern "C" {
  /*** I1 ***/

  double I1(double z2);
  double I1dmu(double z2);

  /*** I2 ***/

  double integrandI2part1(double x, void * params);
  double integrandI2part2(double x, void * params);
  double integralI2Real(double w, double E, double mu, double beta);
  double integralI2Imag(double w, double E, double mu, double beta);

  // TODO: I2dmu
  double integrandI2dmuPart1(double x, void * params);

  /*** T matrix ***/

  double invTmatrixMB_real(double w, void * params);
  double invTmatrixMB_imag(double w, void * params);

  /*** Matsubara sum: pole contribution ***/

  double polePos(double E, double mu, double beta, double a);
  double integrandPoleRes(double x, void * params);
  double integralPoleRes(double E, double mu, double beta, double z0);
  double poleRes(double E, double mu, double beta, double a);
  double poleRes_pole(double E, double mu, double beta, double a, double z0);

  /*** Matsubara sum: branch contribution ***/

  double integrandBranch(double y, void * params);
  double integralBranch(double E, double mu, double beta, double a);

  /*** Density: pole contribution ***/

  double integrandDensityPole(double x, void * params);
  double integralDensityPole(double mu, double beta, double a);

  /*** Density: branch contribution ***/

  double integrandDensityBranch(double x, void * params);
  double integralDensityBranch(double mu, double beta, double a);
}

/*** C++ only ***/

std::complex<double> I2(double w, double E, double mu, double beta);

std::complex<double> invTmatrixMB(double w, double E, double mu, double beta, double a);

