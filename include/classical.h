#pragma once
#include "common.h"

extern "C" {
  double integrandSuscp_cr(double x, void * params);
  double integrandSuscp_ci(double x, void * params);

  double integralSuscp_czr(double w, double mu_ph, double m_i, double m_r);
  double integralSuscp_czi(double w, double mu_ph, double m_i, double m_r);

  double integralSuscp_cr(double w, double E, double mu_ph, double m_i, double m_r);
  double integralSuscp_ci(double w, double E, double mu_ph, double m_i, double m_r);
}
