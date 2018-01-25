#pragma once
#include "common.h"

extern "C" {
  double integrandSuscp_cr(double x, void * params);
  double integrandSuscp_ci(double x, void * params);

  double integralSuscp_czr(double w, double mu_ph, double m_i, double m_r);
  double integralSuscp_czi(double w, double mu_ph, double m_i, double m_r);

  double integralSuscp_cr(double w, double E, double mu_ph, double m_i, double m_r);
  double integralSuscp_ci(double w, double E, double mu_ph, double m_i, double m_r);

  double suscp_cr(double w, double p, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0);
  double suscp_ci(double w, double p, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0);

  double suscp_czr(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0);
  double suscp_czi(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0);

  double Ginv_ph_czr(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0, double E_0, double p_0, double m_ph);
  double Ginv_ph_czi(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0, double E_0, double p_0, double m_ph);
}

std::complex<double> integralSuscp_cc(double w, double p, double mu_ph, double m_i, double m_r);
std::complex<double> integralSuscp_czc(double w, double mu_ph, double m_i, double m_r);

std::complex<double> suscp_cc(double w, double p, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0);
std::complex<double> suscp_czc(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0);

std::complex<double> Ginv_ph_czc(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0, double E_0, double p_0, double m_ph);

