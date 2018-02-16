#include "utils.h"

s_system::s_system(double m_e, double m_h, double eps_r, double T) :
  m_e(m_e),
  m_h(m_h),
  eps_r(eps_r),
  m_p(m_e * m_h / (m_e + m_h)),
  m_pe(m_h / (m_e + m_h)),
  m_ph(m_e / (m_e + m_h)),
  m_sigma((m_e + m_h) * (1.0 / m_e + 1.0 / m_h)),
  T(T),
  beta(f_beta(T)),
  energy_th(f_energy_th(T))
{
  lambda_th = f_lambda_th(beta, m_p);
  m_pT = m_p / energy_th;
}

void s_system::set_temperature(double T) {
  this->T = T;
  beta = f_beta(T);
  energy_th = f_energy_th(T);
  lambda_th = f_lambda_th(beta, m_p);
  m_pT = m_p / energy_th;
}

