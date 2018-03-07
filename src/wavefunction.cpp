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

