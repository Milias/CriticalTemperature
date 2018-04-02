#include "analytic_2d.h"
#include "analytic_2d_utils.h"

double mb_2d_iod(double n, double lambda_s, const system_data & sys) {
  double b_ex{wf_E<0, 2>(lambda_s, sys)};

  if (isnan(b_ex)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double v{2 * std::pow(2 * M_PI, 2) / (n * (sys.m_pe * sys.m_ph / sys.m_sigma)) * std::exp( b_ex / (4 * M_PI) )};

  return y_eq_s(v);
}

double mb_2d_iod_l(double n, double lambda_s, const system_data & sys) {
  double b_ex{wf_E<1, 2>(0, sys) + sys.delta_E / lambda_s};

  double v{2 * std::pow(2 * M_PI, 2) / (n * (sys.m_pe * sys.m_ph / sys.m_sigma)) * std::exp( b_ex / (4 * M_PI) )};

  return y_eq_s(v);
}

