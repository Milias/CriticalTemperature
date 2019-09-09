/*
 * This file contains several functions related to solving
 * the wavefunction in different situations.
 */

#pragma once

#include "wavefunction.h"

#ifndef SWIG

/*
 * This class is used in an ODE to compute the
 * scattering length assuming a Yukawa potential
 * for the interaction between electrons and holes.
 */

template <typename state, uint32_t pot_index = 0>
class wf_bexc_c {
private:
    double pot_lj(double x) const {
        return lj_param_a / std::pow(x, 12) - lj_param_b / std::pow(x, 6);
    }

    constexpr static double (wf_bexc_c<state, pot_index>::*pot_func[])(
        double) const {{&wf_bexc_c<state, pot_index>::pot_lj}};

    constexpr double pot(double x) const {
        return (this->*pot_func[pot_index])(x);
    }

public:
    double lj_param_a{1.0};
    double lj_param_b{1.0};
    const system_data& sys;

    /*
     * Energy for computing eigenvalues.
     * For scattering calculations E = 0.
     */
    double E{0};

    void operator()(const state& y, state& dy, double x) {
        dy[0]                = y[1];
        const double pot_val = pot(x);

        if (x > global_eps) {
            dy[1] = ((pot_val - E) / sys.c_alpha_bexc - 0.25 / (x * x)) * y[0];
        } else {
            dy[1] = 0;
        }
    }
};

template <bool save = false, uint32_t pot_index>
auto wf_bexc_s_t(
    double E, double lj_param_a, double lj_param_b, const system_data& sys) {
    /*
     * Computes the wavefunction for a given E, and returns
     * (u(x), u'(x), x) for x \in (0, x1], or u(x1) if "save"
     * is true.
     *
     * The iterator stops when |u'(x1)| > u'(x0) == 1.
     */

    constexpr uint32_t x1_exp{2};

    state y{{0.0, 1.0}};
    const double r_m{std::pow(lj_param_a * 2 / lj_param_b, 1.0 / 6)};
    double x0{std::pow((1.0 + M_SQRT2), -1.0 / 6) * r_m},
        x1{r_m * (1 << x1_exp)};

    controlled_stepper_type controlled_stepper;

    wf_bexc_c<state, pot_index> wf{lj_param_a, lj_param_b, sys, E};

    if constexpr (save) {
        std::vector<state> f_vec;
        std::vector<double> t_vec;

        for (uint32_t i = 0; i < max_iter; i++) {
            integrate_adaptive(
                controlled_stepper, wf, y, x0, x1, global_eps,
                ode_observer<state>{f_vec, t_vec});

            if (y[1] > 1 || y[1] < -1) {
                break;

            } else {
                x0 = x1;
                x1 = r_m * (1 << (i + 1 + x1_exp));
            }
        }

        return std::make_tuple(f_vec, t_vec);

    } else {
        for (uint32_t i = 0; i < max_iter; i++) {
            integrate_adaptive(controlled_stepper, wf, y, x0, x1, global_eps);

            if (y[1] > 1 || y[1] < -1) {
                break;

            } else if (std::isnan(y[0])) {
                break;

            } else {
                x0 = x1;
                x1 = 1 << (i + 1 + x1_exp);
            }
        }

        return y[0];
    }
}

template <uint32_t pot_index>
auto wf_bexc_s_r_t(
    double rmax,
    double E,
    double lj_param_a,
    double lj_param_b,
    const system_data& sys) {
    /*
     * Computes the wavefunction for a given E, and returns
     * (u(x), u'(x), x) for x \in (0, x1], or u(x1) if "save"
     * is true.
     *
     * The iterator stops when |u'(x1)| > u'(x0) == 1.
     */

    constexpr uint32_t n_steps{256};
    state y{{0.0, 1.0}};
    double x0{
        std::pow(lj_param_b * 0.5 / lj_param_a * (1.0 + M_SQRT2), -1.0 / 6)},
        x1{rmax};

    controlled_stepper_type controlled_stepper;

    wf_bexc_c<state, pot_index> wf{lj_param_a, lj_param_b, sys, E};

    std::vector<state> f_vec;
    std::vector<double> t_vec;

    boost::numeric::odeint::integrate_n_steps(
        controlled_stepper, wf, y, x0, (x1 - x0) / n_steps, n_steps,
        ode_observer<state>{f_vec, t_vec});

    return std::make_tuple(f_vec, t_vec);
}

/*
 * Count wavefunction nodes.
 */

template <uint32_t pot_index = 0>
uint32_t wf_bexc_n_t(
    double E, double lj_param_a, double lj_param_b, const system_data& sys) {
    auto [f_vec, t_vec] =
        wf_bexc_s_t<true, pot_index>(E, lj_param_a, lj_param_b, sys);
    return wf_n(f_vec);
}

/*
 * Computes the groundstate energy.
 */

template <uint32_t pot_index = 0>
double wf_bexc_E_f(double E, void* params) {
    wf_bexc_E_s* s{static_cast<wf_bexc_E_s*>(params)};

    double r =
        wf_bexc_s_t<false, pot_index>(E, s->lj_param_a, s->lj_param_b, s->sys);

    return r;
}

template <uint32_t pot_index = 0, uint32_t N = 0>
double wf_bexc_E_t(
    double E_min,
    double lj_param_a,
    double lj_param_b,
    const system_data& sys) {
    /*
     * Computes the energy of the groundstate, starting
     * from the energy level of a purely Coulomb potential.
     *
     * Using Brent's algorithm.
     */
    constexpr uint32_t local_max_iter{1 << 7};

    // defined in analytic_utils.h
    wf_bexc_E_s params{sys, lj_param_a, lj_param_b};
    double z_min, z_max, z;

    z_min = E_min;

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
    ///*
    double f{1e-1};

    for (uint32_t i = 1, n0 = 0, n = 0; i <= local_max_iter; i++) {
        z_max = z_min * std::pow(1 - f, i);
        n     = wf_bexc_n_t<pot_index>(z_max, lj_param_a, lj_param_b, sys);

        printf(
            "[%d] searching -- n: %d, z: %.14e, %.14e\n", i, n, z_min, z_max);

        if (n == n0 + 1) {
            break;

        } else if (n > n0 + 1) {
            i = 1;
            f *= 0.5;
            z_max = z_min;

        } else if (z_max > -1e-14) {
            return 0.0;

        } else if (i == 1 && n > 0) {
            return std::numeric_limits<double>::quiet_NaN();

        } else {
            z_min = z_max;
        }
    }

    //*/

    gsl_function funct;
    funct.function = &wf_bexc_E_f<pot_index>;
    funct.params   = &params;

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* s            = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < local_max_iter; iter++) {
        status = gsl_root_fsolver_iterate(s);
        z      = gsl_root_fsolver_root(s);
        z_min  = gsl_root_fsolver_x_lower(s);
        z_max  = gsl_root_fsolver_x_upper(s);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
        printf(
            "[%d] iterating -- %.14f (%f, %f), %f\n", iter, z, z_min, z_max,
            funct.function(z, &params));
    }

    gsl_root_fsolver_free(s);
    return z;
}

#endif

std::vector<double> wf_bexc_s(
    double E, double lj_param_a, double lj_param_b, const system_data& sys);
std::vector<double> wf_bexc_s_r(
    double rmax,
    double E,
    double lj_param_a,
    double lj_param_b,
    const system_data& sys);
uint32_t wf_bexc_n(
    double E, double lj_param_a, double lj_param_b, const system_data& sys);
double wf_bexc_E(
    double E_min,
    double lj_param_a,
    double lj_param_b,
    const system_data& sys);
