#include "utils.h"

system_data::system_data(double m_e, double m_h, double eps_r, double T) :
  m_e(m_e * c_m_e),
  m_h(m_h * c_m_e),
  eps_r(eps_r),
  dl_m_e(m_e),
  dl_m_h(m_h),
  m_p(c_m_e * m_e * m_h / (m_e + m_h)),
  m_2p(2.0 * m_p),
  m_pe(m_h / (m_e + m_h)),
  m_ph(m_e / (m_e + m_h)),
  m_sigma((m_e + m_h) * (1.0 / m_e + 1.0 / m_h)),
  T(T),
  beta(f_beta(T)),
  energy_th(f_energy_th(T)),
  zt_len(0.5 * c_hbarc / m_2p)
{
  lambda_th = f_lambda_th(beta, m_p);
  m_pT = m_p / energy_th;
  E_1 = - 0.25 * std::pow(M_SQRT2 * c_aEM / eps_r * std::sqrt(m_pT), 2);
  delta_E = std::pow(2, 1.75) * c_aEM / eps_r * std::sqrt(m_pT * M_PI * c_aEM / eps_r * std::sqrt(m_pT));
}

void system_data::set_temperature(double T) {
  this->T = T;
  beta = f_beta(T);
  energy_th = f_energy_th(T);
  lambda_th = f_lambda_th(beta, m_p);
  m_pT = m_p / energy_th;
}

double system_data::get_z1(double E, double mu_t) const {
  return E / m_sigma - mu_t;
  //return 0.25 * (1 - std::pow(m_ph - m_pe, 2)) * E - mu_t;
}

double system_data::get_E_n(double n) const {
  return E_1 / (n*n);
}

