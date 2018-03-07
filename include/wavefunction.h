/*
 * This file contains several functions related to solving
 * the wavefunction in different situations.
 */

#pragma once

#include "common.h"
#include "templates.h"
#include "wavefunction_utils.h"

typedef std::array<double, 2> state;
typedef boost::numeric::odeint::runge_kutta_cash_karp54<state> error_stepper_type;
typedef boost::numeric::odeint::controlled_runge_kutta<error_stepper_type> controlled_stepper_type;

/*
 * This class is used in an ODE to compute the
 * scattering length assuming a Yukawa potential
 * for the interaction between electrons and holes.
 */

template <typename state, uint32_t pot_index = 0>
class wf_c {
  private:
    double pot_yuk_3d(double x) const {
      return
        - sys.c_aEM / sys.eps_r *
        std::sqrt(2 * sys.m_pT) *
        std::exp(
          -4 * std::sqrt(sys.c_aEM * M_PI / sys.eps_r * std::sqrt(sys.m_pT / 8)) * x / lambda_s
        ) / x;
    }

    constexpr static double (wf_c<state>::*pot_func [])(double) const {{ &wf_c<state>::pot_yuk_3d }};

    double pot(double x) const {
      return (this->*pot_func[pot_index])(x);
    }

  public:
    double lambda_s;
    const system_data & sys;

    /*
     * Energy for computing eigenvalues.
     * For scattering calculations E = 0.
     */
    double E{0};

    void operator()(const state & y, state & dy, double x) {
      double m{std::sqrt(-E)};
      dy[0] = y[1];
      dy[1] = (pot(x) + m / x) * y[0] + (2 * m - 1 / x) * y[1];
    }
};

template <bool save = false, uint32_t pot_index>
auto wf_s(double E, double lambda_s, const system_data & sys) {
  /*
   * Computes the wavefunction for a given E, and returns
   * (u(x), u'(x), x) for x \in (0, x1], or u(x1) if "save"
   * is true.
   *
   * The iterator stops when |u'(x1)| > u'(x0) == 1.
   */

  constexpr uint32_t x1_exp{0};

  state y{{0.0, 1.0}};
  double x0{1e-10}, x1{1<<x1_exp};

  controlled_stepper_type controlled_stepper;
  wf_c<state, pot_index> wf{lambda_s, sys, E};

  if constexpr(save) {
    std::vector<state> f_vec;
    std::vector<double> t_vec;

    for (uint32_t i = 0; i < max_iter; i++) {
      integrate_adaptive(controlled_stepper, wf, y, x0, x1, global_eps, ode_observer<state>{f_vec, t_vec});

      if (y[1] > 1 || y[1] < -1) {
        break;
      } else {
        x0 = x1;
        x1 = 1<<(i+1+x1_exp);
      }
    }

    return std::make_tuple(f_vec, t_vec);

  } else {
    for (uint32_t i = 0; i < max_iter; i++) {
      integrate_adaptive(controlled_stepper, wf, y, x0, x1, global_eps);

      if (y[1] > 1 || y[1] < -1) {
        break;
      } else {
        x0 = x1;
        x1 = 1<<(i+1+x1_exp);
      }
    }

    return y[0];
  }
}

/*
 * Count wavefunction nodes.
 */

uint32_t wf_n(const std::vector<state> & f_vec);

template <uint32_t pot_index>
uint32_t wf_n(double E, double lambda_s, const system_data & sys) {
  auto [f_vec, t_vec] = wf_s<true, pot_index>(E, lambda_s, sys);
  return wf_n(f_vec);
}

/*
 * Computes the groundstate energy.
 */

template <uint32_t pot_index>
double wf_E_f(double E, void * params) {
  wf_E_s * s{static_cast<wf_E_s*>(params)};

  return wf_s<false, pot_index>(E, s->lambda_s, s->sys);
}

template <uint32_t pot_index>
double wf_E(double lambda_s, const system_data & sys) {
  /*
   * Computes the energy of the groundstate, starting
   * from the energy level of a purely Coulomb potential.
   *
   * Using Brent's algorithm.
   */

  // defined in analytic_utils.h
  wf_E_s params{lambda_s, sys};
  double z_min{sys.E_1}, z_max, z;

  /*
   * "f" is the factor the original energy gets reduced
   * by on each iteration.
   *
   * If the change on nodes is equal to 1, then break
   * the loop.
   *
   * If the change is larger, it means that the step is
   * too large, so reduce it and go back to the previous
   * step.
   */
  double f{1e-1};
  for (uint32_t i = 0, n0 = 0, n = 0; i < max_iter; i++) {
    z_max = z_min * std::pow(1 - f, i);
    n = wf_n<pot_index>(z_max, lambda_s, sys);

    //printf("n: %d, z: %f, %f\n", n, z_min, z_max);

    if (n == n0 + 1) {
      break;
    } else if (n > n0 + 1) {
      f *= 0.5;
      i = 0;
      z_max = z_min;
    } else if (- z_max < global_eps) {
      return std::numeric_limits<double>::quiet_NaN();
    } else {
      z_min = z_max;
    }
  }

  gsl_function funct;
  funct.function = &wf_E_f<pot_index>;
  funct.params = &params;

  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);

  gsl_root_fsolver_set(s, &funct, z_min, z_max);

  for (int status = GSL_CONTINUE, iter = 0; status == GSL_CONTINUE && iter < max_iter; iter++) {
    status = gsl_root_fsolver_iterate(s);
    z = gsl_root_fsolver_root(s);
    z_min = gsl_root_fsolver_x_lower(s);
    z_max = gsl_root_fsolver_x_upper(s);

    status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    //printf("%f, %f, %f\n", lambda_s, z, funct.function(z, &params));
  }

  gsl_root_fsolver_free(s);
  return z;
}

