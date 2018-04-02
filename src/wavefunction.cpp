#include "wavefunction.h"

uint32_t wf_n(const std::vector<state> & f_vec) {
  /*
   * Computes the number of nodes of the given wavefunction
   * by iterating and increasing the counter when two successive
   * points change sign.
   */

  uint32_t nodes{0};

  for (uint32_t i = 1; i < f_vec.size(); i++) {
    if (f_vec[i][0] * f_vec[i - 1][0] < 0) {
      nodes++;
    }

    if (std::abs(f_vec[i][1]) > f_vec[0][1]) {
      break;
    }
  }

  return nodes;
}

std::vector<double> wf_s_py(double E, double lambda_s, const system_data & sys) {
  auto [f_vec, t_vec] = wf_s<true, 0, 3>(E, lambda_s, sys);
  std::vector<double> r(3 * f_vec.size());

  for (uint32_t i = 0; i < f_vec.size(); i++) {
    r[3 * i] = f_vec[i][0];
    r[3 * i + 1] = f_vec[i][1];
    r[3 * i + 2] = t_vec[i];
  }

  return r;
}

std::vector<double> wf_2d_s_py(double E, double lambda_s, const system_data & sys) {
  auto [f_vec, t_vec] = wf_s<true, 0, 2>(E, lambda_s, sys);
  std::vector<double> r(3 * f_vec.size());

  for (uint32_t i = 0; i < f_vec.size(); i++) {
    r[3 * i] = f_vec[i][0];
    r[3 * i + 1] = f_vec[i][1];
    r[3 * i + 2] = t_vec[i];
  }

  return r;
}

uint32_t wf_n_py(double E, double lambda_s, const system_data & sys) {
  return wf_n<0, 3>(E, lambda_s, sys);
}

uint32_t wf_2d_n_py(double E, double lambda_s, const system_data & sys) {
  return wf_n<0, 2>(E, lambda_s, sys);
}

double wf_E_py(double lambda_s, const system_data & sys) {
  return wf_E<0, 3>(lambda_s, sys);
}

double wf_2d_E_py(double lambda_s, const system_data & sys) {
  return wf_E<0, 2>(lambda_s, sys);
}

uint32_t wf_n_cou_py(double E, const system_data & sys) {
  return wf_n<1, 3>(E, 0, sys);
}

uint32_t wf_2d_n_cou_py(double E, const system_data & sys) {
  return wf_n<1, 2>(E, 0, sys);
}

double wf_E_cou_py(const system_data & sys) {
  return wf_E<1, 3>(0, sys);
}

double wf_2d_E_cou_py(const system_data & sys) {
  return wf_E<1, 2>(0, sys);
}

