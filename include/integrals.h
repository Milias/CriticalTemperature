#pragma once

#include "common.h"

/*** I1 ***/

double I1(double z2);
double I1dmu(double z2);

/*** I2 ***/

double integrandI2part1(double x, void * params);
double integrandI2part2(double x, void * params);
double integralI2Real(double w, double E, double mu, double beta);
double integralI2Imag(double w, double E, double mu, double beta);

std::complex<double> I2(double w, double E, double mu, double beta);

/*** T matrix ***/

double invTmatrixMB_real(double w, void * params);
std::complex<double> invTmatrixMB(double w, double E, double mu, double beta, double a);

double polePos(double E, double mu, double beta, double a);
double integrandPoleRes(double x, void * params);
double integralPoleRes(double E, double mu, double beta, double z0);
double poleRes(double E, double mu, double beta, double a);
double poleRes(double E, double mu, double beta, double a, double z0);

double integrandBranch(double y, void * params);
double integralBranch(double E, double mu, double beta, double a);

