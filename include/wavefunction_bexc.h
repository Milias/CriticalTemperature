/*
 * This file contains several functions related to solving
 * the wavefunction in different situations.
 */

#pragma once

#include "wavefunction.h"

#ifndef SWIG

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

    constexpr uint32_t x1_exp{1};

    state y{{0.0, 1.0}};
    double x0{rmin}, x1{rmax == 0.0 ? rmin * (1 << x1_exp) : rmax};

    controlled_stepper_type controlled_stepper;

    wf_dy_s<state, pot_s> wf{alpha, E, pot};

    if constexpr (save) {
        std::vector<state> f_vec;
        std::vector<double> t_vec;

        for (uint32_t i = 0; i < max_iter; i++) {
            integrate_adaptive(
                controlled_stepper, wf, y, x0, x1, global_eps,
                ode_observer<state>{f_vec, t_vec});

            if (rmax > 0.0) {
                break;
            }

            if (y[1] > 1 || y[1] < -1) {
                break;

            } else {
                x0 = x1;
                x1 = rmin * (1 << (i + 1 + x1_exp));
            }
        }

        return std::make_tuple(f_vec, t_vec);

    } else {
        for (uint32_t i = 0; i < max_iter; i++) {
            integrate_adaptive(controlled_stepper, wf, y, x0, x1, global_eps);

            if (rmax > 0.0) {
                break;
            }

            if (y[1] > 1 || y[1] < -1) {
                break;

            } else if (std::isnan(y[0])) {
                break;

            } else {
                x0 = x1;
                x1 = rmin * (1 << (i + 1 + x1_exp));
            }
        }

        return y[0];
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
    double x0{rmin}, x1{rmax};

    controlled_stepper_type controlled_stepper;

    wf_dy_s<state, pot_s> wf{alpha, E, pot};

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

template <class pot_s>
uint32_t wf_gen_n_t(
    double E, double rmin, double alpha, pot_s& pot, double rmax = 0.0) {
    auto [f_vec, t_vec] = wf_gen_s_t<true, pot_s>(E, rmin, alpha, pot, rmax);
    return wf_n(f_vec);
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
    constexpr uint32_t local_max_iter{1 << 7};

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
    ///*
    double f{1e-1};

    for (uint32_t i = 1, n0 = 0, n = 0; i <= local_max_iter; i++) {
        z_max = z_min * std::pow(1 - f, i);
        n     = wf_gen_n_t<pot_s>(
            z_max, params.rmin, params.alpha, params.pot, params.rmax);

        /*
        printf(
            "[%d] searching -- n: %d, z: %.14e, %.14e\n", i, n, z_min, z_max);
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

    //*/

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

        //status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
        status = gsl_root_test_residual(funct.function(z, &params), global_eps);
        /*
        printf(
            "[%d] iterating -- %.16f (%f, %f), %f\n", iter, z, z_min, z_max,
            funct.function(z, &params));
        */
    }

    gsl_root_fsolver_free(s);
    return z;
}

#endif
