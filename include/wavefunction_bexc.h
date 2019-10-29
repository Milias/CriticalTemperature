/*
 * This file contains several functions related to solving
 * the wavefunction in different situations.
 */

#pragma once

#include "common.h"
#include "templates.h"
#include "wavefunction_utils.h"

#ifndef SWIG

typedef std::array<double, 2> state;
typedef boost::numeric::odeint::runge_kutta_fehlberg78<state>
    error_stepper_type_gen;
typedef boost::numeric::odeint::controlled_runge_kutta<error_stepper_type_gen>
    controlled_stepper_type_gen;

template <bool save = false, class pot_s>
auto wf_gen_s_t(
    double E, double rmin, double alpha, pot_s& pot, double rmax = 0.0) {
    /*
     * Computes the wavefunction for a given E, and returns
     * (u(x), u'(x), x) for x \in (0, x1], or u(x1) if "save"
     * is true.
     *
     * The iterator stops when |u'(x1)| > u'(x0) == 1.
     */

    state y{{0.0, 1.0}};
    double x0{rmin > 0 ? rmin : 1e-20}, x1{rmax > 0 ? rmax : 2};

    constexpr uint32_t local_max_iter{1 << 6};
    constexpr double local_eps{1e-16};

    controlled_stepper_type_gen controlled_stepper;

    wf_dy_s<state, pot_s> wf{alpha, E, pot};

    bool continue_integ{true};

    if constexpr (save) {
        std::vector<state> f_vec;
        std::vector<double> t_vec;
        std::vector<state> full_f_vec;
        std::vector<double> full_t_vec;

        for (uint32_t i = 0; i < local_max_iter; i++) {
            integrate_adaptive(
                controlled_stepper, wf, y, x0, x1, local_eps,
                ode_observer<state>{f_vec, t_vec});

            if (std::isfinite(y[0])) {
                full_f_vec.insert(
                    full_f_vec.end(), f_vec.begin(), f_vec.end());
                full_t_vec.insert(
                    full_t_vec.end(), t_vec.begin(), t_vec.end());

                f_vec.clear();
                t_vec.clear();

                x1 *= 2;
            } else {
                break;
            }
        }

        return std::make_tuple(full_f_vec, full_t_vec);
    } else {
        state new_y{{y[0], y[1]}};
        for (uint32_t i = 0; i < local_max_iter; i++) {
            integrate_adaptive(
                controlled_stepper, wf, y, x0, x1, local_eps);

            if (std::isfinite(y[0])) {
                new_y = y;

                x1 *= 2;
            } else {
                break;
            }
        }

        return new_y[0];
    }
}

template <class pot_s>
auto wf_gen_s_r_t(
    double E,
    double rmin,
    double alpha,
    double rmax,
    uint32_t n_steps,
    pot_s& pot) {
    state y{{0.0, 1.0}};
    double x0{rmin > 0 ? rmin : 1e-20}, x1{rmax};

    controlled_stepper_type_gen controlled_stepper;

    wf_dy_s<state, pot_s> wf{alpha, E, pot};

    std::vector<state> f_vec;
    std::vector<double> t_vec;

    ///*
    boost::numeric::odeint::integrate_n_steps(
        controlled_stepper, wf, y, x0, (x1 - x0) / n_steps, n_steps,
        ode_observer<state>{f_vec, t_vec});
    //*/

    /*
    integrate_adaptive(
        controlled_stepper, wf, y, x0, x1, 1e-12,
        ode_observer<state>{f_vec, t_vec});
    */

    return std::make_tuple(f_vec, t_vec);
}

/*
 * Count wavefunction nodes.
 */

uint32_t wf_gen_n(const std::vector<state>& f_vec);

template <class pot_s>
uint32_t wf_gen_n_t(
    double E, double rmin, double alpha, pot_s& pot, double rmax = 0.0) {
    auto [f_vec, t_vec] = wf_gen_s_t<true, pot_s>(E, rmin, alpha, pot, rmax);
    return wf_gen_n(f_vec);
}

/*
 * Computes the groundstate energy.
 */

template <class pot_s>
double wf_gen_E_f(double E, wf_gen_E_s<pot_s>* s) {
    return wf_gen_s_t<false, pot_s>(E, s->rmin, s->alpha, s->pot, s->rmax);
}

template <class pot_s, uint32_t N = 0>
double wf_gen_E_t(
    double E_min, double rmin, double alpha, pot_s& pot, double rmax = 0.0) {
    /*
     * Computes the energy of the groundstate, starting
     * from the energy level of a purely Coulomb potential.
     *
     * Using Brent's algorithm.
     */
    constexpr uint32_t local_max_iter{1 << 6};

    // defined in analytic_utils.h
    wf_gen_E_s<pot_s> params{rmin, alpha, pot, rmax};
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
    double f{1e-1};

    for (uint32_t i = 1, n0 = 0, n = 0; i <= local_max_iter; i++) {
        z_max = z_min * std::pow(1 - f, i);

        n = wf_gen_n_t<pot_s>(
            z_max, params.rmin, params.alpha, params.pot, params.rmax);

        /*
        printf(
            "[%d] searching -- n: %d, z: %.14e, %.14e\n", i, n, z_min,
        z_max);
        */

        if (i == 1 && n > 0) {
            return std::numeric_limits<double>::quiet_NaN();

        } else if (n == n0 + 1) {
            break;

        } else if (n > n0 + 1) {
            i = 1;
            f *= 0.5;
            z_max = z_min;

        } else if (z_max > -1e-14) {
            return 0.0;

        } else {
            z_min = z_max;
        }
    }

    gsl_function funct;
    funct.function = &templated_f<wf_gen_E_s<pot_s>, wf_gen_E_f<pot_s>>;
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

        status = gsl_root_test_interval(z_min, z_max, 0, 1e-14);

        ///*
        printf(
            "[%d] iterating -- (%f, %f), (%.2e, %.2e)\n", iter, z_min, z_max,
            funct.function(z_min, &params), funct.function(z_max, &params));
        //*/
    }

    gsl_root_fsolver_free(s);
    return z;
}

#endif
