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

double integralSuscp_ci(double w, double E, double mu_ph, double m_i, double m_r) {
  double r = 0;

  if (E < 1e-8) {
    r = integralSuscp_czi(w, mu_ph, m_i, m_r);
  } else {
    //double params_arr[] = {w, E, mu_ph, m_i, m_r};
    r = 0;
  }

  return r;
}

double integralSuscp_czr(double w, double mu_ph, double m_i, double m_r) {
  return integralSuscp_czc(w, mu_ph, m_i, m_r).real();
}

double integralSuscp_czi(double w, double mu_ph, double m_i, double m_r) {
  return integralSuscp_czc(w, mu_ph, m_i, m_r).imag();
}

std::complex<double> integralSuscp_czc(double w, double mu_ph, double m_i, double m_r) {
  std::complex<double> z{- m_r / (4 * M_PI * m_i) * std::complex<double>(mu_ph, w)};
  std::complex<double> sqrt_val{std::sqrt(z)};
  std::complex<double> exp_val{std::exp(z)};
  std::complex<double> erfc_val{1.0 - erf_c(sqrt_val)};

  return 0.5 - 0.5 * std::sqrt(M_PI) * (sqrt_val * exp_val * erfc_val);
}

std::complex<double> integralSuscp_cc(double w, double p, double mu_ph, double m_i, double m_r) {
  /*
   * p is hbar * k
   *
   * */

  if (p < 1e-6) {
    return integralSuscp_czc(w, mu_ph, m_i, m_r);
  }

  return 0;
}

double suscp_cr(double w, double p, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0) {
  return suscp_cc(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0).real();
}

double suscp_ci(double w, double p, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0) {
  return suscp_cc(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0).imag();
}

double suscp_czr(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0) {
  return suscp_czc(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0).real();
}

double suscp_czi(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0) {
  return suscp_czc(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0).imag();
}

std::complex<double> suscp_cc(double w, double p, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0) {
  /*
   * p is hbar * k
   * V_0 is V_0 / hbar^2
   *
   * */
  double mu_ph = mu_1 + mu_2;

  // Ek is in units of E_th = 1/4/pi/beta
  double Ek1 = 0.5 * p*p / m_1, Ek2 = 0.5 * p*p / m_2, Ekr = 0.5 * p*p / m_r;

  std::complex<double> z(mu_ph, w);
  std::complex<double> sqrt_val{0.5 * M_PI * std::sqrt(0.25 * Ekr - z)};
  std::complex<double> int1_val{std::sqrt(0.5 * m_1 / m_r) * std::exp(- beta * (0.25 * Ek1 - mu_1)) * integralSuscp_cc(w, p, mu_ph, m_1, m_r)};
  std::complex<double> int2_val{std::sqrt(0.5 * m_2 / m_r) * std::exp(- beta * (0.25 * Ek2 - mu_2)) * integralSuscp_cc(w, p, mu_ph, m_2, m_r)};

  std::complex<double> total_val{0.5 * m_r * (sqrt_val + int1_val + int2_val)};

  return -total_val / ( 1.0 + V_0 * total_val );
}

std::complex<double> suscp_czc(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0) {
  /*
   * p is hbar * k
   * V_0 is V_0 / hbar^2
   *
   * */
  double mu_ph = mu_1 + mu_2;

  std::complex<double> z(mu_ph, w);
  std::complex<double> sqrt_val{0.5 * M_PI * std::sqrt(- z)};
  std::complex<double> int1_val{std::sqrt(0.5 * m_1 / m_r) * std::exp(beta * mu_1) * integralSuscp_czc(w, mu_ph, m_1, m_r)};
  std::complex<double> int2_val{std::sqrt(0.5 * m_2 / m_r) * std::exp(beta * mu_2) * integralSuscp_czc(w, mu_ph, m_2, m_r)};

  std::complex<double> total_val{0.5 * m_r * (sqrt_val + int1_val + int2_val)};

  return -total_val / ( 1.0 + V_0 * total_val );
}

std::complex<double> Ginv_ph_czc(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0, double E_0, double p_0, double m_ph) {
  double E_ph = E_0 + 0.5 * m_ph * p_0*p_0;

  return std::complex<double>(E_ph - mu_1 - mu_2, -w) - suscp_czc(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0);
}

double Ginv_ph_czr(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0, double E_0, double p_0, double m_ph) {
  return Ginv_ph_czc(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0, E_0, p_0, m_ph).real();
}

double Ginv_ph_czi(double w, double mu_1, double mu_2, double m_1, double m_2, double m_r, double beta, double V_0, double E_0, double p_0, double m_ph) {
  return Ginv_ph_czc(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0, E_0, p_0, m_ph).imag();
}

