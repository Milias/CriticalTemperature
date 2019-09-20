/*
 * This file contains several functions related to solving
 * the wavefunction in different situations.
 */

#pragma once

#include "wavefunction.h"

#ifndef SWIG

template <bool save = false, double (*pot)(double, const system_data&)>
auto wf_gen_s_t(double E, double rmin, double alpha, const system_data& sys) {
    /*
     * Computes the wavefunction for a given E, and returns
     * (u(x), u'(x), x) for x \in (0, x1], or u(x1) if "save"
     * is true.
     *
     * The iterator stops when |u'(x1)| > u'(x0) == 1.
     */

    constexpr uint32_t x1_exp{2};

    state y{{0.0, 1.0}};
    double x0{rmin}, x1{1 << x1_exp};

    controlled_stepper_type controlled_stepper;

    wf_dy_s<state, pot> wf{alpha, E, sys};

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
                x1 = 1 << (i + 1 + x1_exp);
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

template <double (*pot)(double, const system_data&)>
auto wf_gen_s_r_t(
    double E, double rmin, double alpha, double rmax, const system_data& sys) {
    /*
     * Computes the wavefunction for a given E, and returns
     * (u(x), u'(x), x) for x \in (0, x1], or u(x1) if "save"
     * is true.
     *
     * The iterator stops when |u'(x1)| > u'(x0) == 1.
     */

    constexpr uint32_t n_steps{256};
    state y{{0.0, 1.0}};
    double x0{rmin}, x1{rmax};

    controlled_stepper_type controlled_stepper;

    wf_dy_s<state, pot> wf{alpha, E, sys};

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

template <double (*pot)(double, const system_data&)>
uint32_t wf_gen_n_t(
    double E, double rmin, double alpha, const system_data& sys) {
    auto [f_vec, t_vec] = wf_gen_s_t<true, pot>(E, rmin, alpha, sys);
    return wf_n(f_vec);
}

/*
 * Computes the groundstate energy.
 */

template <double (*pot)(double, const system_data&)>
double wf_gen_E_f(double E, wf_gen_E_s* s) {
    return wf_gen_s_t<false, pot>(E, s->rmin, s->alpha, s->sys);
}

template <double (*pot)(double, const system_data&), uint32_t N = 0>
double wf_gen_E_t(
    double E_min, double rmin, double alpha, const system_data& sys) {
    /*
     * Computes the energy of the groundstate, starting
     * from the energy level of a purely Coulomb potential.
     *
     * Using Brent's algorithm.
     */
    constexpr uint32_t local_max_iter{1 << 7};

    // defined in analytic_utils.h
    wf_gen_E_s params{rmin, alpha, sys};
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
        n     = wf_gen_n_t<pot>(z_max, params.rmin, params.alpha, params.sys);

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
    funct.function = &templated_f<wf_gen_E_s, wf_gen_E_f<pot>>;
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
