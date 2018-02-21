/*
 * Here we define several helper structures
 * used in analytic.cpp.
 */

#pragma once

#include "common.h"
#include "analytic.h"

struct analytic_n_sc_s {
  double mu_t;
  double a;
};

struct ideal_mu_v_s {
  double v;
  const system_data & sys;
};

/*
 * This class is used in an ODE to compute the
 * scattering length assuming a Yukawa potential
 * for the interaction between electrons and holes.
 */
template <typename T>
class analytic_a_n_s {
  private:
    double pot(double x) {
      return - sys.c_aEM / sys.eps_r * std::sqrt(2 * sys.m_pT) * std::exp(-4 * std::sqrt(sys.c_aEM * M_PI / sys.eps_r * std::sqrt(sys.m_pT / 8)) * x / lambda_s) / x;
    }

  public:
    double lambda_s;
    const system_data & sys;

    void operator()(const T & y, T & dy, double x) {
      dy[0] = y[1];
      dy[1] = pot(x) * y[0];
    }
};

struct analytic_mu_s {
  double n;
  const system_data & sys;
};

