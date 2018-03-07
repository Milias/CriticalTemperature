/*
 * Here we define several helper structures
 * used in analytic.cpp.
 */

#pragma once

#include "common.h"

struct analytic_n_sc_s {
  double mu_t;
  double a;
};

struct ideal_mu_v_s {
  double v;
  const system_data & sys;
};

typedef std::array<double, 2> state;
typedef boost::numeric::odeint::runge_kutta_cash_karp54<state> error_stepper_type;
typedef boost::numeric::odeint::controlled_runge_kutta<error_stepper_type> controlled_stepper_type;

/*
 * This class is used in an ODE to compute the
 * scattering length assuming a Yukawa potential
 * for the interaction between electrons and holes.
 */

template <typename T>
class analytic_a_n_s {
  private:
    double pot(double x) {
      return
        - sys.c_aEM / sys.eps_r *
        std::sqrt(2 * sys.m_pT) *
        std::exp(
          -4 * std::sqrt(sys.c_aEM * M_PI / sys.eps_r * std::sqrt(sys.m_pT / 8)) * x / lambda_s
        ) / x;
    }

  public:
    double lambda_s;
    const system_data & sys;

    /*
     * Energy for computing eigenvalues.
     * For scattering calculations E = 0.
     */
    double E{0};

    void operator()(const T & y, T & dy, double x) {
      dy[0] = y[1];
      dy[1] = (pot(x) - E) * y[0];
    }
};

/*
 * Struct used when saving the full wavefunction.
 */

template <typename state, typename time = double>
struct ode_observer {
  std::vector<state> & f_vec;
  std::vector<time> & t_vec;

  void operator()(const state & f, time t) {
    f_vec.push_back(f);
    t_vec.push_back(t);
  }
};

struct analytic_mu_s {
  double n;
  const system_data & sys;
};

struct analytic_b_ex_s {
  double lambda_s;
  const system_data & sys;
};
