#include "classical.h"

double integrandSuscp_cr(double x, void * params) {
  double * params_arr = (double*)params;
  double w = params_arr[0];
  double E = params_arr[1];
  double mu_ph = params_arr[2];
  double m_i = params_arr[3];
  double m_r = params_arr[4];

  return (x + 0.25 * E - mu_ph) * std::exp(- x * m_r / (4 * M_PI * m_i)) * std::sinh(std::sqrt(m_r * E * x / m_i) / (4 * M_PI)) / ( w*w + std::pow(x + 0.25 * E - mu_ph, 2) );
}

double integrandSuscp_ci(double x, void * params) {
  double * params_arr = (double*)params;
  double w = params_arr[0];
  double E = params_arr[1];
  double mu_ph = params_arr[2];
  double m_i = params_arr[3];
  double m_r = params_arr[4];

  return w * std::exp(- x * m_r / (4 * M_PI * m_i)) * std::sinh(std::sqrt(m_r * E * x / m_i) / (4 * M_PI)) / ( w*w + std::pow(x + 0.25 * E - mu_ph, 2) );
}

double integralSuscp_cr(double w, double E, double mu_ph, double m_i, double m_r) {
  double r = 0;

  if (E < 1e-8) {
    r = integralSuscp_czr(w, mu_ph, m_i, m_r);
  } else {
    //double params_arr[] = {w, E, mu_ph, m_i, m_r};
    r = 0;
  }

  return r;
}

double integralSuscp_czr(double w, double mu_ph, double m_i, double m_r) {
  return 0;
}

double integralSuscp_czi(double w, double mu_ph, double m_i, double m_r) {
  return 0;
}

double integralSuscp_ci(double w, double E, double mu_ph, double m_i, double m_r) {
  return 0;
}


