#include "plasmons.h"
#include "plasmons_utils.h"

template <
    typename T,
    uint8_t remove_term = 0,
    bool invert         = true,
    bool include_cou    = true>
T plasmon_green(
    double w,
    double k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12) {
    const double k2{k * k};

    const double E[2] = {
        sys.m_pe * sys.c_alpha * k2,
        sys.m_ph * sys.c_alpha * k2,
    };

    T w_complex;

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        w_complex = {w, delta};
    } else {
        w_complex = w;
    }

    const T nu[4] = {
        -w_complex / E[0] - 1.0,
        -w_complex / E[0] + 1.0,
        -w_complex / E[1] - 1.0,
        -w_complex / E[1] + 1.0,
    };

    T pi_screen_nofactor[2];

    if constexpr (remove_term == 0) {
        pi_screen_nofactor[0] =
            -2.0 -
            nu[0] * std::sqrt(1.0 - 4.0 * mu_e / (E[0] * nu[0] * nu[0])) +
            nu[1] * std::sqrt(1.0 - 4.0 * mu_e / (E[0] * nu[1] * nu[1]));

        pi_screen_nofactor[1] =
            -2.0 -
            nu[2] * std::sqrt(1.0 - 4.0 * mu_h / (E[1] * nu[2] * nu[2])) +
            nu[3] * std::sqrt(1.0 - 4.0 * mu_h / (E[1] * nu[3] * nu[3]));

    } else if constexpr (remove_term == 1) {
        pi_screen_nofactor[0] =
            -2.0 -
            nu[0] * std::sqrt(1.0 - 4.0 * mu_e / (E[0] * nu[0] * nu[0]));
        /*
         * When m_e < m_h, the argument of the sqrt is zero (or very close),
         * so it has to be removed to avoid complex numbers because of
         * numerical precision.
         */

        pi_screen_nofactor[1] =
            -2.0 -
            nu[2] * std::sqrt(1.0 - 4.0 * mu_h / (E[1] * nu[2] * nu[2])) +
            nu[3] * std::sqrt(1.0 - 4.0 * mu_h / (E[1] * nu[3] * nu[3]));
    }

    T result;

    if constexpr (include_cou) {
        result = -sys.eps_r * k / (sys.c_hbarc * sys.c_aEM) +
                 0.125 * M_1_PI / sys.c_alpha *
                     (pi_screen_nofactor[0] / sys.m_pe +
                      pi_screen_nofactor[1] / sys.m_ph);
    } else {
        result = 0.125 * M_1_PI / sys.c_alpha *
                 (pi_screen_nofactor[0] / sys.m_pe +
                  pi_screen_nofactor[1] / sys.m_ph);
    }

    if constexpr (invert) {
        return 1.0 / result;
    } else {
        return result;
    }
}

template <typename T, bool invert = true, bool include_cou = true>
T plasmon_green_ht(
    double w,
    double k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12) {
    const double k2{k * k};

    const double E[2] = {
        sys.m_pe * sys.c_alpha * k2,
        sys.m_ph * sys.c_alpha * k2,
    };

    T pi_screen_nofactor[2];

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        const T w_complex = {w, delta};

        const T nu[4] = {
            -w_complex - E[0],
            -w_complex + E[0],
            -w_complex - E[1],
            -w_complex + E[1],
        };

        const T nu_exp[2] = {
            -0.25 * (std::pow(w_complex, 2) + std::pow(E[0], 2)) / E[0],
            -0.25 * (std::pow(w_complex, 2) + std::pow(E[1], 2)) / E[1],
        };

        pi_screen_nofactor[0] =
            std::exp(sys.beta * mu_e) / (std::sqrt(sys.beta * E[0]) * M_PI) *
            (std::complex<double>(0, M_SQRTPI) * std::exp(nu_exp[0]) *
                 std::sinh(0.5 * sys.beta * w_complex) -
             Faddeeva::Dawson(std::sqrt(0.25 * sys.beta / E[0]) * nu[0]) +
             Faddeeva::Dawson(std::sqrt(0.25 * sys.beta / E[0]) * nu[1]));

        pi_screen_nofactor[1] =
            std::exp(sys.beta * mu_h) / (std::sqrt(sys.beta * E[1]) * M_PI) *
            (std::complex<double>(0, M_SQRTPI) * std::exp(nu_exp[1]) *
                 std::sinh(0.5 * sys.beta * w_complex) -
             Faddeeva::Dawson(std::sqrt(0.25 * sys.beta / E[1]) * nu[2]) +
             Faddeeva::Dawson(std::sqrt(0.25 * sys.beta / E[1]) * nu[3]));
    } else {
        pi_screen_nofactor[0] =
            2.0 * std::exp(sys.beta * mu_e) /
            (std::sqrt(sys.beta * E[0]) * M_PI) *
            Faddeeva::Dawson(std::sqrt(0.25 * sys.beta * E[0]));

        pi_screen_nofactor[1] =
            2.0 * std::exp(sys.beta * mu_h) /
            (std::sqrt(sys.beta * E[1]) * M_PI) *
            Faddeeva::Dawson(std::sqrt(0.25 * sys.beta * E[1]));
    }

    T result;

    if constexpr (include_cou) {
        result = -sys.eps_r * k / (sys.c_hbarc * sys.c_aEM) -
                 0.25 / sys.c_alpha *
                     (pi_screen_nofactor[0] / sys.m_pe +
                      pi_screen_nofactor[1] / sys.m_ph);
    } else {
        result = -0.25 / sys.c_alpha *
                 (pi_screen_nofactor[0] / sys.m_pe +
                  pi_screen_nofactor[1] / sys.m_ph);
    }

    if constexpr (invert) {
        return 1.0 / result;
    } else {
        return result;
    }
}

template <bool invert = true>
double plasmon_green_lwl(
    double _w,
    double k,
    double ls,
    double _mu_h,
    const system_data& sys,
    double delta = 1e-12) {
    if constexpr (invert) {
        return 1.0 / (-sys.eps_r * (k + ls) / (sys.c_hbarc * sys.c_aEM));
    } else {
        return -sys.eps_r * (k + ls) / (sys.c_hbarc * sys.c_aEM);
    }
}

std::vector<std::complex<double>> plasmon_green_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_total{wk_vec.size()};
    using T = std::complex<double>;

    std::vector<T> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] = plasmon_green<T>(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
    }

    return result;
}

std::vector<std::complex<double>> plasmon_green_inv_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_total{wk_vec.size()};
    using T = std::complex<double>;

    std::vector<T> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] =
            plasmon_green<T, 0, false>(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
    }

    return result;
}

std::vector<std::complex<double>> plasmon_green_ht_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_total{wk_vec.size()};
    using T = std::complex<double>;

    std::vector<T> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] =
            plasmon_green_ht<T>(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
    }

    return result;
}

std::vector<std::complex<double>> plasmon_green_ht_inv_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_total{wk_vec.size()};
    using T = std::complex<double>;

    std::vector<T> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] =
            plasmon_green_ht<T, false>(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
    }

    return result;
}

double plasmon_kmax_f(double k, void* params) {
    using T = std::complex<double>;
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    double w{s->sys.m_pe * k * k + 2.0 * std::sqrt(s->sys.m_pe * s->mu_e) * k};

    return plasmon_green<T, 1, false>(w, k, s->mu_e, s->mu_h, s->sys).real();
}

double plasmon_kmax(double mu_e, double mu_h, const system_data& sys) {
    /*
     * Evaluates the Green's function at
     *   w = m_pe * k^2 + 2 * sqrt(mu_e * m_pe) * k
     * and searches for the point at which Ginv is zero.
     *
     * This point is when the plasmon pole disappears.
     * Assumes k > 1e-5.
     * TODO: what about k < 1e-5?
     *
     * First perform an exponential sweep to find the upper
     * bound.
     * TODO: improve this? Analytic upper bound?
     */

    plasmon_potcoef_s s{0, 0, 0, mu_e, mu_h, sys};

    double z{0};

    double z_min{1e-5}, z_max{1.0};

    gsl_function funct;
    funct.function = &plasmon_kmax_f;
    funct.params   = &s;

    /*
     * Expontential sweep
     */

    const uint32_t max_pow{20};
    double ginv_upper{0}, upper_bound{z_max};

    for (uint32_t ii = 1; ii <= max_pow; ii++) {
        ginv_upper = funct.function(upper_bound, funct.params);

        if (ginv_upper < 0) {
            z_max = upper_bound;
            break;

        } else if (ginv_upper == 0) {
            return z_max;

        } else {
            z_min       = upper_bound;
            upper_bound = z_max * (1 << ii);
        }
    }

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver       = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_wmax(
    double mu_e, double mu_h, double v_1, const system_data& sys) {
    double kmax{plasmon_kmax(mu_e, mu_h, sys)};
    return sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax;
}

double plasmon_wmax(double kmax, double mu_e, const system_data& sys) {
    return sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax;
}

double plasmon_disp_f(double w, void* params) {
    using T = std::complex<double>;
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    return plasmon_green<T, 0, false>(w, s->k1, s->mu_e, s->mu_h, s->sys)
        .real();
}

template <bool check_bounds>
double plasmon_disp_t(
    double k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double kmax = 0) {
    if (k == 0) {
        return 0;
    }

    else if (k < 0) {
        k = -k;
    }

    if constexpr (check_bounds) {
        kmax = plasmon_kmax(mu_e, mu_h, sys);

        if (k > kmax) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double z{0};
    double z_min{sys.m_pe * k * k + 2 * std::sqrt(sys.m_pe * mu_e) * k},
        z_max{sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax};

    z_min = z_min + (z_max - z_min) * 1e-5;

    plasmon_potcoef_s s{0, k, 0, mu_e, mu_h, sys};

    gsl_function funct;
    funct.function = &plasmon_disp_f;
    funct.params   = &s;

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver       = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_disp(
    double k, double mu_e, double mu_h, const system_data& sys) {
    return plasmon_disp_t<true>(k, mu_e, mu_h, sys);
}

double plasmon_disp_ncb(
    double k, double mu_e, double mu_h, const system_data& sys, double kmax) {
    return plasmon_disp_t<false>(k, mu_e, mu_h, sys, kmax);
}

double plasmon_disp_inv_f(double k, void* params) {
    using T = std::complex<double>;
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    std::complex<double> result{
        plasmon_green<T, 0, false>(s->w, k, s->mu_e, s->mu_h, s->sys)};

    return result.real();
}

template <bool check_bounds>
double plasmon_disp_inv_t(
    double w, double mu_e, double mu_h, const system_data& sys) {
    /*
     * Essentially same as plasmon_disp, but it computes k(w), the inverse
     * dispersion relation.
     */

    if (w == 0) {
        return 0;
    } else if (w < 1e-5) {
        return w;
    } else if (w < 0) {
        w = -w;
    }

    if constexpr (check_bounds) {
        double kmax{plasmon_kmax(mu_e, mu_h, sys)};
        double w_max{sys.m_pe * kmax * kmax +
                     2 * std::sqrt(sys.m_pe * mu_e) * kmax};

        if (w > w_max) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double z, z_min{1e-5},
        z_max{std::sqrt(2 * sys.m_e * mu_e) * (std::sqrt(1 + w / mu_e) - 1) /
              sys.c_hbarc};

    plasmon_potcoef_s s{w, 0, 0, mu_e, mu_h, sys};

    gsl_function funct;
    funct.function = &plasmon_disp_inv_f;
    funct.params   = &s;

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver       = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_disp_inv(
    double w, double mu_e, double mu_h, const system_data& sys) {
    return plasmon_disp_inv_t<true>(w, mu_e, mu_h, sys);
}

double plasmon_disp_inv_ncb(
    double w, double mu_e, double mu_h, const system_data& sys) {
    return plasmon_disp_inv_t<false>(w, mu_e, mu_h, sys);
}

template <
    std::complex<double> (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N = 0>
double plasmon_disp_th_f(double th, void* params) {
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    double k{
        std::sqrt(
            s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)),
    };

    std::complex<double> result{
        green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta),
    };

    return result.real();
}

double plasmon_disp_th(
    const double wkk[3], double mu_e, double mu_h, const system_data& sys) {
    double w{wkk[0]};

    // if (std::abs(wkk[0]) < 1e-10) {
    if (wkk[0] == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (wkk[0] < 0) {
        w = -wkk[0];
    }

    double k2_max{
        2 * sys.m_e * mu_e *
            std::pow((sqrt(1 + w / mu_e) - 1) / sys.c_hbarc, 2),
    };

    if (k2_max > std::pow(wkk[1] + wkk[2], 2) ||
        k2_max < std::pow(wkk[1] - wkk[2], 2)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double th_max{std::acos(
        (k2_max - wkk[1] * wkk[1] - wkk[2] * wkk[2]) /
        (-2 * wkk[1] * wkk[2]))};

    double z, z_min{th_max * 1e-5}, z_max{th_max * (1 - 1e-5)};

    plasmon_potcoef_s s{w, wkk[1], wkk[2], mu_e, mu_h, sys};

    gsl_function funct;
    funct.function =
        &plasmon_disp_th_f<plasmon_green<std::complex<double>, 0, false>>;
    funct.params = &s;

    if (funct.function(z_max, funct.params) < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver       = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

template <
    typename T,
    uint8_t return_part,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N = 0>
auto plasmon_potcoef_f(double th, void* params) {
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    double k{
        std::sqrt(
            s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)),
    };

    if constexpr (N == 0) {
        T green{green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta)};

        if constexpr (return_part == 0 || std::is_same<T, double>::value) {
            return green;
        } else if constexpr (return_part == 1) {
            return green.real();
        } else if constexpr (return_part == 2) {
            return green.imag();
        }
    } else {
        std::complex<double> green{
            /*
            green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta) *
                std::exp(std::complex<double>(0, N * th)),
            */
            green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta) *
                std::complex<double>(std::cos(N * th), std::sin(N * th)),
        };

        if constexpr (return_part == 0) {
            return green;
        } else if constexpr (return_part == 1) {
            return green.real();
        } else if constexpr (return_part == 2) {
            return green.imag();
        }
    }
}

template <
    typename T,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool find_pole = true,
    uint32_t N     = 0>
auto plasmon_potcoef(
    const double wkk[3],
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    constexpr uint32_t n_int{2};
    double result[n_int] = {0}, error[n_int] = {0};

    /*
     * wkk -> (w, k1, k2)
     */

    double k2{wkk[2]};

    if (std::abs(wkk[1] - wkk[2]) < 1e-6) {
        k2 *= 1.0 + 1e-5;
    }

    plasmon_potcoef_s s{wkk[0], wkk[1], k2, mu_e, mu_h, sys, delta};

    gsl_function integrands[n_int];

    integrands[0].function = &plasmon_potcoef_f<T, 1, green_func, N>;
    integrands[0].params   = &s;

    constexpr uint32_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-8};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    if constexpr (
        find_pole && std::is_same<T, std::complex<double>>::value && N == 0) {
        double th_pts[3] = {
            M_PI,
            plasmon_disp_th(wkk, mu_e, mu_h, sys),
            0,
        };

        if (!std::isnan(th_pts[1])) {
            gsl_integration_qagp(
                integrands, th_pts, 3, global_eps, 0, local_ws_size, ws,
                result, error);
        } else {
            gsl_integration_qag(
                integrands, M_PI, 0, global_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result, error);
        }
    } else {
        if constexpr (N == 0) {
            gsl_integration_qag(
                integrands, M_PI, 0, local_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result, error);
        } else {
            gsl_integration_qag(
                integrands, 2 * M_PI, 0, local_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result, error);

            result[0] *= 0.5;
        }
    }

    if constexpr (std::is_same<T, std::complex<double>>::value || N > 0) {
        integrands[1].function = &plasmon_potcoef_f<T, 2, green_func, N>;
        integrands[1].params   = &s;

        if constexpr (N == 0) {
            gsl_integration_qag(
                integrands + 1, 0, M_PI, global_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result + 1, error + 1);
        } else {
            gsl_integration_qag(
                integrands + 1, 0, 2 * M_PI, global_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result + 1, error + 1);

            result[1] *= 0.5;
        }
    }

    gsl_integration_workspace_free(ws);

    /*
    for (uint32_t i = 0; i < n_int; i++) {
        printf("%s [%d]: %f (%e)\n", __func__, i, result[i], error[i]);
    }
    printf("\n");
    */

    if constexpr (std::is_same<T, std::complex<double>>::value || N > 0) {
        return std::complex<double>(result[0] * M_1_PI, result[1] * M_1_PI);
    } else {
        return result[0] * M_1_PI;
    }
}

std::vector<std::complex<double>> plasmon_potcoef_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    using T = std::complex<double>;

    plasmon_mat_s<T, plasmon_potcoef<T, plasmon_green>> s(
        N_k, N_w, sys, delta);

    s.fill_mat_potcoef(mu_e, mu_h);

    std::vector<T> result_vec(s.mat_potcoef.n_elem);

#pragma omp parallel for
    for (uint32_t i = 0; i < s.mat_potcoef.n_elem; i++) {
        result_vec[i] = s.mat_potcoef(i);
    }

    return result_vec;
}

std::vector<std::complex<double>> plasmon_potcoef_ht_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    const std::complex<double>& z,
    double delta) {
    using T = std::complex<double>;

    plasmon_mat_s<T, plasmon_potcoef<T, plasmon_green>> s(
        N_k, N_w, sys, delta);
    s.fill_mat_potcoef(mu_e, mu_h);

    if (!std::isnan(z.real())) {
        s.fill_mat_elem();
        s.update_mat_potcoef(z);
    }

    std::vector<T> result_vec(s.mat_potcoef.n_elem);

#pragma omp parallel for
    for (uint32_t i = 0; i < s.mat_potcoef.n_elem; i++) {
        result_vec[i] = s.mat_potcoef(i);
    }

    return result_vec;
}

template <uint8_t return_part = 0, typename T, typename T2, uint32_t N>
auto plasmon_det_f(T2 z, void* params) {
    plasmon_mat_s<T, plasmon_potcoef<T, plasmon_green, false, N>>* s{
        static_cast<
            plasmon_mat_s<T, plasmon_potcoef<T, plasmon_green, false, N>>*>(
            params),
    };

    s->update_mat_potcoef(z);

    if constexpr (return_part == 0) {
        return arma::det(s->mat_potcoef);
    } else if constexpr (return_part == 1) {
        return arma::det(s->mat_potcoef).real();
    } else if constexpr (return_part == 2) {
        return arma::det(s->mat_potcoef).imag();
    }
}

template <
    typename T,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool sweep    = true,
    uint32_t N    = 0,
    bool const_eb = false>
double plasmon_det_zero_t(
    double mu_e,
    double mu_h,
    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>& s,
    double eb_min = std::numeric_limits<double>::quiet_NaN()) {
    if constexpr (const_eb) {
        return s.sys.get_E_n(N + 0.5);
    }

    constexpr double local_eps{1e-8};
    double z{-s.sys.get_E_n(N + 0.5)};
    double z_min{std::isnan(eb_min) ? z : -eb_min}, z_max{z};

    if (z + eb_min < local_eps) {
        return -z;
    }

    s.fill_mat_potcoef(mu_e, mu_h);
    s.fill_mat_elem();

    gsl_function funct;
    if constexpr (std::is_same<T, std::complex<double>>::value || N > 0) {
        funct.function = &plasmon_det_f<1, T, double, N>;
    } else {
        funct.function = &plasmon_det_f<0, T, double, N>;
    }
    funct.params = &s;

    if constexpr (sweep) {
        const uint32_t max_pow{20};
        double f_val{0};
        bool return_nan{true};

        for (uint32_t ii = 1; ii <= max_pow; ii++) {
            f_val = funct.function(z_min, funct.params);

            if (f_val < 0) {
                return_nan = false;
                break;

            } else if (f_val == 0) {
                return z_max;

            } else {
                z_max = z_min;
                z_min *= 0.7;
            }
        }

        if (return_nan) {
            if (mu_e > 0) {
                return 0.0;
            }

            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        /*
        printf(
            "[%s] eb_min: %f, max: %.16f, min: %.16f\n", __func__, eb_min,
            z_max, z_min);
        */

        status = gsl_root_test_interval(z_min, z_max, 0, local_eps);
    }

    gsl_root_fsolver_free(solver);
    return -z;
}

double plasmon_det_zero(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> s(
        N_k, 1, sys, delta);

    if (std::isnan(eb_min)) {
        return det_zero(mu_e, mu_h, s, eb_min);
    } else {
        return det_zero_f(mu_e, mu_h, s, eb_min);
    }
}

double plasmon_det_zero_ht(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> s(
        N_k, 1, sys, delta);

    if (std::isnan(eb_min)) {
        return det_zero(mu_e, mu_h, s, eb_min);
    } else {
        return det_zero_f(mu_e, mu_h, s, eb_min);
    }
}

std::vector<double> plasmon_det_zero_v(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = std::complex<double>;
    constexpr auto green_func = plasmon_green<T>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> s(
        N_k, 1, sys, delta);

    uint64_t N{mu_vec.size()};
    std::vector<double> result(N, 0.0);

    for (uint32_t i = 0; i < N; i++) {
        result[i] = plasmon_det_zero_t<T, green_func, false>(
            mu_vec[i], sys.get_mu_h_t0(mu_vec[i]), s, eb_min);

        if (result[i] == 0) {
            break;
        }
    }

    return result;
}

std::vector<double> plasmon_det_zero_v1(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = std::complex<double>;
    constexpr auto green_func = plasmon_green<T>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 1>> s(
        N_k, 1, sys, delta);

    uint64_t N{mu_vec.size()};
    std::vector<double> result(N, 0.0);

    for (uint32_t i = 0; i < N; i++) {
        result[i] = plasmon_det_zero_t<T, green_func, true, 1>(
            mu_vec[i], sys.get_mu_h_t0(mu_vec[i]), s, eb_min);

        if (result[i] == 0) {
            break;
        }
    }

    return result;
}

std::vector<double> plasmon_det_zero_ht_v(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> s(
        N_k, 1, sys, delta);

    uint64_t N{mu_vec.size()};
    std::vector<double> result(N, 0.0);

    for (uint32_t i = 0; i < N; i++) {
        result[i] = plasmon_det_zero_t<T, green_func>(
            mu_vec[i], sys.get_mu_h(mu_vec[i]), s, eb_min);

        if (result[i] == 0) {
            break;
        }
    }

    return result;
}

std::vector<double> plasmon_det_zero_ht_v1(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = std::complex<double>;
    constexpr auto green_func = plasmon_green_ht<T>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 1>> s(
        N_k, 1, sys, delta);

    uint64_t N{mu_vec.size()};
    std::vector<double> result(N, 0.0);

    for (uint32_t i = 0; i < N; i++) {
        result[i] = plasmon_det_zero_t<T, green_func, true, 1>(
            mu_vec[i], sys.get_mu_h(mu_vec[i]), s, eb_min);

        if (result[i] == 0) {
            break;
        }
    }

    return result;
}

double plasmon_det_zero_lwl(uint32_t N_k, double ls, const system_data& sys) {
    using T = double;

    plasmon_mat_s<T, plasmon_potcoef<T, plasmon_green_lwl, false, 0>> s(
        N_k, 1, sys, 0);
    return plasmon_det_zero_t<T, plasmon_green_lwl>(ls, 0, s);
}

double plasmon_det_zero_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = std::complex<double>;
    constexpr auto green_func = plasmon_green<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> s(
        N_k, N_w, sys, delta);

    if (std::isnan(eb_min)) {
        return det_zero(mu_e, mu_h, s, eb_min);
    } else {
        return det_zero_f(mu_e, mu_h, s, eb_min);
    }
}

double plasmon_det_zero_ht_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min,
    double delta) {
    using T                   = std::complex<double>;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> s(
        N_k, N_w, sys, delta);

    if (std::isnan(eb_min)) {
        return det_zero(mu_e, mu_h, s, eb_min);
    } else {
        return det_zero_f(mu_e, mu_h, s, eb_min);
    }
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool exact,
    bool include_tail = false>
double plasmon_rpot_f(double k, void* params) {
    plasmon_rpot_s* s{static_cast<plasmon_rpot_s*>(params)};

    double elem;

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        elem = green_func(0, k, s->mu_e, s->mu_h, s->sys, s->delta).real();
    } else {
        elem = green_func(0, k, s->mu_e, s->mu_h, s->sys, s->delta);
    }

    if constexpr (exact) {
        if constexpr (include_tail) {
            return k * elem * gsl_sf_bessel_J0(k * s->x);
        } else {
            return (k * elem - s->sys.c_aEM * s->sys.c_hbarc / s->sys.eps_r) *
                   gsl_sf_bessel_J0(k * s->x);
        }
    } else {
        if constexpr (include_tail) {
            return elem * std::sqrt(k * M_1_PI / s->x);
        } else {
            return (k * elem - s->sys.c_aEM * s->sys.c_hbarc / s->sys.eps_r) /
                   std::sqrt(M_PI * k * s->x);
        }
    }
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool include_tail = false>
double plasmon_rpot_t(
    double x,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12) {
    constexpr uint32_t n_int{2};
    constexpr uint32_t local_ws_size{1 << 9};
    constexpr uint32_t t_size{1 << 9};

    double result[n_int] = {0}, error[n_int] = {0};
    constexpr uint32_t n_sum{1 << 7};

    double result_sum = {0.0};
    double last_zero  = {0.0};
    double t[n_sum];

    plasmon_rpot_s s{x, mu_e, mu_h, sys};

    gsl_function integrands[n_int];

    integrands[0].function =
        &plasmon_rpot_f<T, green_func, true, include_tail>;
    integrands[0].params = &s;

    integrands[1].function =
        &plasmon_rpot_f<T, green_func, false, include_tail>;
    integrands[1].params = &s;

    gsl_integration_qawo_table* qawo_table[2] = {
        gsl_integration_qawo_table_alloc(x, 1, GSL_INTEG_COSINE, t_size),
        gsl_integration_qawo_table_alloc(x, 1, GSL_INTEG_SINE, t_size),
    };

    gsl_integration_workspace* ws[2] = {
        gsl_integration_workspace_alloc(local_ws_size),
        gsl_integration_workspace_alloc(local_ws_size),
    };

    for (uint32_t i = 0; i < n_sum; i++) {
        double temp{gsl_sf_bessel_zero_J0(i + 1) / x};

        gsl_integration_qag(
            integrands, last_zero, temp, 0.0, global_eps, local_ws_size,
            GSL_INTEG_GAUSS31, ws[0], result, error);

        t[i]      = result[0];
        last_zero = temp;
    }

    gsl_sum_levin_u_workspace* w = gsl_sum_levin_u_alloc(n_sum);
    gsl_sum_levin_u_accel(t, n_sum, w, &result_sum, error);
    gsl_sum_levin_u_free(w);

    gsl_integration_qawf(
        integrands + 1, last_zero, 0, t_size, ws[0], ws[1], qawo_table[0],
        result + 1, error + 1);

    result_sum += result[1];

    gsl_integration_qawf(
        integrands + 1, last_zero, 0, t_size, ws[0], ws[1], qawo_table[1],
        result + 1, error + 1);

    result_sum += result[1];

    gsl_integration_workspace_free(ws[0]);
    gsl_integration_workspace_free(ws[1]);

    gsl_integration_qawo_table_free(qawo_table[0]);
    gsl_integration_qawo_table_free(qawo_table[1]);

    if constexpr (include_tail) {
        return result_sum;
    } else {
        return sys.c_aEM * sys.c_hbarc / (x * sys.eps_r) + result_sum;
    }
}

std::vector<double> plasmon_rpot_v(
    const std::vector<double>& x_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_x{x_vec.size()};
    std::vector<double> output(N_x);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_x; i++) {
        output[i] = plasmon_rpot_t<std::complex<double>, plasmon_green>(
            x_vec[i], mu_e, mu_h, sys);
    }

    return output;
}

std::vector<double> plasmon_rpot_ht_v(
    const std::vector<double>& x_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_x{x_vec.size()};
    std::vector<double> output(N_x);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_x; i++) {
        output[i] = plasmon_rpot_t<double, plasmon_green_ht>(
            x_vec[i], mu_e, mu_h, sys);
    }

    return output;
}

std::vector<double> plasmon_rpot_lwl_v(
    const std::vector<double>& x_vec,
    double ls,
    const system_data& sys,
    double delta) {
    uint64_t N_x{x_vec.size()};
    std::vector<double> output(N_x);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_x; i++) {
        output[i] =
            plasmon_rpot_t<double, plasmon_green_lwl>(x_vec[i], ls, 0, sys);
    }

    return output;
}

double plasmon_rpot(
    double x, double mu_e, double mu_h, const system_data& sys) {
    return plasmon_rpot_t<std::complex<double>, plasmon_green>(
        x, mu_e, mu_h, sys);
}

double plasmon_rpot_ht(
    double x, double mu_e, double mu_h, const system_data& sys) {
    return plasmon_rpot_t<double, plasmon_green>(x, mu_e, mu_h, sys);
}

double plasmon_exc_mu_zero_f(double x, void* params) {
    plasmon_exc_mu_zero_s* s{static_cast<plasmon_exc_mu_zero_s*>(params)};

    return x - std::pow(1 + x, 1 - s->sys.m_eh);
}

double plasmon_exc_mu_zero(const system_data& sys) {
    plasmon_exc_mu_zero_s s{sys};

    gsl_function funct;
    funct.function = &plasmon_exc_mu_zero_f;
    funct.params   = &s;

    double z{0};
    double z_min{1}, z_max{2};

    /*
     * Expontential sweep
     */

    const uint32_t max_pow{20};
    double f_val{0}, f_val_prev{0};
    bool return_nan{true};

    f_val_prev = funct.function(z_min, funct.params);
    if (f_val_prev == 0) {
        return 0.0;
    }

    for (uint32_t ii = 0; ii < max_pow; ii++) {
        f_val = funct.function(z_max, funct.params);

        if (f_val * f_val_prev < 0) {
            return_nan = false;
            break;

        } else if (f_val == 0) {
            return std::log(z_max) / sys.beta;

        } else {
            f_val_prev = f_val;
            z_min      = z_max;
            z_max      = 1 << (ii + 2);
        }
    }

    if (return_nan) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);

    return std::log(z) / sys.beta;
}

double plasmon_exc_mu_val_f(double x, void* params) {
    plasmon_exc_mu_zero_s* s{static_cast<plasmon_exc_mu_zero_s*>(params)};

    return x + s->sys.get_mu_h(x) - s->val;
}

double plasmon_exc_mu_val_df(double x, void* params) {
    plasmon_exc_mu_zero_s* s{static_cast<plasmon_exc_mu_zero_s*>(params)};

    double result;

    if (s->sys.beta * x < -20) {
        result = 2;
    } else if (s->sys.beta * x > 20) {
        result = 1 + s->sys.m_eh;
    } else {
        result =
            1 +
            (s->sys.m_eh * std::exp(s->sys.beta * x) *
             std::pow(1 + std::exp(s->sys.beta * x), s->sys.m_eh - 1)) /
                (s->sys.beta *
                 (std::pow(1 + std::exp(s->sys.beta * x), s->sys.m_eh) - 1));
    }

    return result;
}

void plasmon_exc_mu_val_fdf(double x, void* params, double* y, double* dy) {
    *y  = plasmon_exc_mu_val_f(x, params);
    *dy = plasmon_exc_mu_val_df(x, params);
}

double plasmon_exc_mu_val(double val, const system_data& sys) {
    if (val == 0) {
        return plasmon_exc_mu_zero(sys);
    }

    plasmon_exc_mu_zero_s s{sys, val};

    gsl_function_fdf funct;
    funct.f      = &plasmon_exc_mu_val_f;
    funct.df     = &plasmon_exc_mu_val_df;
    funct.fdf    = &plasmon_exc_mu_val_fdf;
    funct.params = &s;

    double z{
        std::max(
            0.5 * (val - std::log(sys.m_eh) / sys.beta), val / (1 + sys.m_eh)),
    };
    double z0{0};

    if (std::abs(funct.f(z, &s)) < 1e-8) {
        return z;
    }

    const gsl_root_fdfsolver_type* solver_type = gsl_root_fdfsolver_steffenson;
    gsl_root_fdfsolver* solver = gsl_root_fdfsolver_alloc(solver_type);

    gsl_root_fdfsolver_set(solver, &funct, z);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < 128; iter++) {
        z0     = z;
        status = gsl_root_fdfsolver_iterate(solver);
        z      = gsl_root_fdfsolver_root(solver);
        status = gsl_root_test_residual(funct.f(z, &s), 1e-8);
    }

    gsl_root_fdfsolver_free(solver);

    return z;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
double plasmon_exc_mu_lim_f(double mu_e, void* params) {
    plasmon_exc_mu_lim_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>* s{
        static_cast<plasmon_exc_mu_lim_s<
            plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>*>(
            params),
    };

    double mu_h{s->mat_s.sys.get_mu_h(mu_e)};

    return mu_e + mu_h + s->val - det_zero(mu_e, mu_h, s->mat_s, s->eb_lim);
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
double plasmon_exc_mu_lim_t(
    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>& mat_s,
    double val    = 0.0,
    double mu_lim = std::numeric_limits<double>::quiet_NaN(),
    double eb_lim = std::numeric_limits<double>::quiet_NaN()) {
    if (val == 0 && !std::isnan(mu_lim) && false /* TODO: change */) {
        /*
         * In this case we have a trivial solution, because we
         * have already computed it.
         */
        return mu_lim;
    }

    plasmon_exc_mu_lim_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>
        s{
            mat_s,
            val,
            eb_lim,
        };

    gsl_function funct;
    funct.function = &plasmon_exc_mu_lim_f<T, green_func, N, det_zero>;
    funct.params   = &s;

    double z;
    double z_min{
        s.mat_s.sys.get_E_n(0.5) -
            std::log(s.mat_s.sys.m_eh) / s.mat_s.sys.beta - val,
    };

    double z_max{
        val == 0 || std::isnan(mu_lim) ? plasmon_exc_mu_zero(s.mat_s.sys)
                                       : mu_lim,
    };

    /*
    printf(
        "[%s] z: (%.10f, %.10f), f: (%f, %f)\n", __func__, z_max, z_min,
        funct.function(z_max, &s), funct.function(z_min, &s));
    */

    const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        // printf("[%s] z: (%.10f, %.10f)\n", __func__, z_max, z_min);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
double plasmon_exc_mu_lim_int_f(double mu_e, void* params) {
    plasmon_exc_mu_lim_int_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>* s{
        static_cast<plasmon_exc_mu_lim_s<
            plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>*>(
            params),
    };

    double mu_h{s->mat_s.sys.get_mu_h(mu_e)};
    double eb{det_zero(mu_e, mu_h, s->mat_s, s->eb_lim)};
    double mu_exc{eb - mu_e - mu_h};

    double v0n{
        4 * M_PI * s->sys.c_hbarc * s->sys.c_hbarc /
            ((s->sys.m_e + s->sys.m_h) *
             std::log(
                 4 * s->sys.c_hbarc * s->sys.c_hbarc /
                 (mu_exc * (s->sys.m_e + s->sys.m_h) * s->sys.a0 *
                  s->sys.a0))) *
            s->sys.density_exc_exp(s->u),
    };

    return mu_exc + v0n - s->val;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
double plasmon_exc_mu_lim_int_t(
    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>& mat_s,
    double u,
    double mu_lim = std::numeric_limits<double>::quiet_NaN(),
    double eb_lim = std::numeric_limits<double>::quiet_NaN()) {
    const double val{logExp(u) / mat_s.sys.beta};

    plasmon_exc_mu_lim_int_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>
        s{
            mat_s,
            u,
            val,
            eb_lim,
        };

    gsl_function funct;
    funct.function = &plasmon_exc_mu_lim_int_f<T, green_func, N, det_zero>;
    funct.params   = &s;

    double z;
    double z_min{
        s.mat_s.sys.get_E_n(0.5) -
            std::log(s.mat_s.sys.m_eh) / s.mat_s.sys.beta - val,
    };

    double z_max{
        val == 0 || std::isnan(mu_lim) ? plasmon_exc_mu_zero(s.mat_s.sys)
                                       : mu_lim,
    };

    /*
    printf(
        "[%s] z: (%.10f, %.10f), f: (%f, %f)\n", __func__, z_max, z_min,
        funct.function(z_max, &s), funct.function(z_min, &s));
    */

    const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        // printf("[%s] z: (%.10f, %.10f)\n", __func__, z_max, z_min);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_exc_mu_lim_ht(
    uint32_t N_k, const system_data& sys, double val, double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func, true, 0>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> mat_s(
        N_k, 1, sys, delta);
    return plasmon_exc_mu_lim_t<T, green_func, 0, det_zero>(mat_s, val);
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
std::tuple<double, double, double> plasmon_mu_e_u(
    double u,
    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>* s) {
    double mu_e{0.0}, mu_h{0.0}, eb{0.0};

    if (u > -15) {
        mu_e = plasmon_exc_mu_lim_t<T, green_func, N, det_zero>(
            s->mat_s, logExp(u) / s->mat_s.sys.beta, s->mu_e_lim, s->eb_lim);

        // mu_h = s->mat_s.sys.get_mu_h_ht(mu_e);
        // eb = det_zero(mu_e, mu_h, s->mat_s, s->eb_lim);
    } else {
        mu_e = s->mu_e_lim;
        // mu_h = s->mat_s.sys.get_mu_h_ht(mu_e);
        // eb   = s->eb_lim;
    }

    return {mu_e, mu_h, eb};
}

template <
    typename T,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double),
    bool substract_total,
    typename TS = void>
double plasmon_density_mu_f(double u, TS* params) {
    /*
     * beta * (eb - mu_ex) = log(1 + exp(u))
     */
    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>* s;
    if constexpr (std::is_same<TS, void>::value) {
        s = static_cast<plasmon_density_s<
            plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>*>(
            params);
    } else {
        s = params;
    }

    auto [mu_e, mu_h, eb] = plasmon_mu_e_u<T, green_func, N, det_zero>(u, s);

    /*
    printf(
        "[%s] mu_e_lim: %f, eb_lim: %f, u: %f, %.16f, %.16f\n", __func__,
        s->mu_e_lim, s->eb_lim, u, mu_e, eb);
    */

    if constexpr (substract_total) {
        return s->mat_s.sys.density_ideal(mu_e) +
               s->mat_s.sys.density_exc_exp(u) - s->n_total;
        // s->mat_s.sys.density_exc(mu_e + mu_h, eb) - s->n_total;
    } else {
        return s->mat_s.sys.density_ideal(mu_e) +
               s->mat_s.sys.density_exc_exp(u);
        // s->mat_s.sys.density_exc(mu_e + mu_h, eb);
    }
}

template <
    typename T,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double),
    bool substract_total,
    typename TS = void>
double plasmon_density_mu_df(double u, TS* params) {
    constexpr auto f =
        plasmon_density_mu_f<T, green_func, N, det_zero, substract_total, TS>;
    constexpr double h{1e-5};

    return (f(u + h, params) - f(u - h, params)) * 0.5 / h;
}

template <
    typename T,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double),
    bool substract_total,
    typename TS = void>
void plasmon_density_mu_fdf(double u, TS* params, double* y, double* dy) {
    y[0] =
        plasmon_density_mu_f<T, green_func, N, det_zero, substract_total, TS>(
            u, params);
    dy[0] =
        plasmon_density_mu_df<T, green_func, N, det_zero, substract_total, TS>(
            u, params);
}

std::vector<double> plasmon_density_mu_ht_v(
    const std::vector<double>& u_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func, true>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> mat_s(
        N_k, 1, sys, delta);
    double mu_e_lim{plasmon_exc_mu_lim_t<T, green_func, 0, det_zero>(mat_s)};
    double eb_lim{
        det_zero(
            mu_e_lim, sys.get_mu_h(mu_e_lim), mat_s,
            std::numeric_limits<double>::quiet_NaN()),
    };

    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>>>
        s{
            mat_s,
            mu_e_lim,
            eb_lim,
        };

    uint64_t N_z{u_vec.size()};

    std::vector<double> result(N_z);

    printf("Progress: %u/%lu\n", 0, N_z);
    for (uint32_t i = 0, c = 0; i < N_z; i++, c++) {
        if (10 * c >= N_z) {
            printf("Progress: %u/%lu\n", i + 1, N_z);
            c = 0;
        }

        result[i] = plasmon_density_mu_f<T, green_func, 0, det_zero_f, false>(
            u_vec[i], &s);
    }
    printf("Progress: %lu/%lu\n", N_z, N_z);

    return result;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
double plasmon_density_t(
    double n_total,
    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>& s) {
    s.n_total = n_total;

    gsl_function funct;
    funct.function = &plasmon_density_mu_f<T, green_func, N, det_zero, true>;
    funct.params   = &s;

    double z{0};
    double z_max{s.mat_s.sys.mu_exc_u(n_total)};
    double z_min;

    /*
     * For the sweep:
     * if z_max < 0: exponential growing sweep.
     * if z_max > 0: take z_max -> -z_max and exponential growing sweep.
     * if z_max ~ 0: take z_max -> -1 ?
     */

    if (z_max > 0) {
        z_min = -z_max;
    } else if (std::abs(z_max) < 1e-6) {
        z_min = -1;
    } else {
        z_min = 1.5 * z_max;
    }

    double f_min;
    bool return_nan{true};
    uint32_t max_sweep_iter{64};
    for (uint32_t i = 0; i < max_sweep_iter; i++) {
        f_min = funct.function(z_min, &s);

        if (f_min > 0) {
            return_nan = false;
            break;
        } else {
            z_max = z_min;
            z_min *= 2;
        }
    }

    // printf("[%s] min: %.16f, max: %.16f\n", __func__, z_min, z_max);

    if (return_nan) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    /*
    double f_max{funct.function(z_max, &s)};

    printf(
        "[%s] min: %.16f, max: %.16f, f: (%.3e, %.3e)\n", __func__, z_min,
        z_max, f_min, f_max);
    */

    const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        // printf("[%s] min: %.16f, max: %.16f\n", __func__, z_min, z_max);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);

    return plasmon_exc_mu_lim_t<T, green_func, det_zero>(
        s.mat_s, logExp(z) / s.mat_s.sys.beta, s.mu_e_lim, s.eb_lim);
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
std::tuple<double, double> plasmon_density_ts(
    double n_total,
    double z_init,
    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>& s) {
    s.n_total = n_total;

    double z, z0;

    if (std::isnan(z_init)) {
        z = s.mat_s.sys.mu_exc_u(n_total);
    } else {
        z = z_init;
    }

    gsl_function_fdf funct;
    funct.f      = &plasmon_density_mu_f<T, green_func, N, det_zero, true>;
    funct.df     = &plasmon_density_mu_df<T, green_func, N, det_zero, true>;
    funct.fdf    = &plasmon_density_mu_fdf<T, green_func, N, det_zero, true>;
    funct.params = &s;

    const gsl_root_fdfsolver_type* solver_type = gsl_root_fdfsolver_steffenson;
    gsl_root_fdfsolver* solver = gsl_root_fdfsolver_alloc(solver_type);

    gsl_root_fdfsolver_set(solver, &funct, z);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < 32; iter++) {
        double f_z{funct.f(z, &s)};
        z0     = z;
        status = gsl_root_fdfsolver_iterate(solver);
        z      = gsl_root_fdfsolver_root(solver);
        status = gsl_root_test_residual(f_z, 1e-10);

        printf(
            "[%s] iter: %d, z: %f, z0: %f, f_z: %e\n", __func__, iter, z, z0,
            f_z);
    }

    gsl_root_fdfsolver_free(solver);

    return {
        z,
        plasmon_exc_mu_lim_t<T, green_func, N, det_zero>(
            s.mat_s, logExp(z) / s.mat_s.sys.beta, s.mu_e_lim, s.eb_lim),
    };
}

std::vector<double> plasmon_density_ht_v(
    const std::vector<double>& n_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func, true>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> mat_s(
        N_k, 1, sys, delta);

    double mu_e_lim{plasmon_exc_mu_lim_t<T, green_func, 0, det_zero>(mat_s)};
    double eb_lim{
        det_zero(
            mu_e_lim, sys.get_mu_h(mu_e_lim), mat_s,
            std::numeric_limits<double>::quiet_NaN()),
    };

    printf("[%s] mu_e: %f, eb: %f\n", __func__, mu_e_lim, eb_lim);

    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>>>
        s{
            mat_s,
            mu_e_lim,
            eb_lim,
        };

    uint64_t N{n_vec.size() + 2};
    std::vector<double> result(N);
    std::vector<double> u_vec(N);

    result[0] = mu_e_lim;
    result[1] = eb_lim;

    printf("Progress: %u/%lu\n", 0, N);
    for (uint32_t i = 2, c = 0; i < N; i++, c++) {
        if (10 * c >= N) {
            printf("Progress: %u/%lu\n", i + 1, N);
            c = 0;
        }

        auto [u, mu_e] = plasmon_density_ts<T, green_func, 0, det_zero_f>(
            n_vec[i - 2],
            i > 2 ? u_vec[i - 1] : std::numeric_limits<double>::quiet_NaN(),
            s);

        u_vec[i]  = u;
        result[i] = mu_e;
    }
    printf("Progress: %lu/%lu\n", N, N);

    return result;
}

std::vector<double> plasmon_density_ht_c_v(
    const std::vector<double>& n_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero = plasmon_det_zero_t<T, green_func, true, 0, true>;
    constexpr auto det_zero_f =
        plasmon_det_zero_t<T, green_func, false, 0, true>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> mat_s(
        N_k, 1, sys, delta);

    double mu_e_lim{plasmon_exc_mu_lim_t<T, green_func, 0, det_zero>(mat_s)};
    double eb_lim{
        det_zero(
            mu_e_lim, sys.get_mu_h(mu_e_lim), mat_s,
            std::numeric_limits<double>::quiet_NaN()),
    };

    printf("[%s] mu_e: %f, eb: %f\n", __func__, mu_e_lim, eb_lim);

    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>>>
        s{
            mat_s,
            mu_e_lim,
            eb_lim,
        };

    uint64_t N{n_vec.size() + 2};
    std::vector<double> result(N);
    std::vector<double> u_vec(N);

    result[0] = mu_e_lim;
    result[1] = eb_lim;

    printf("Progress: %u/%lu\n", 0, N);
    for (uint32_t i = 2, c = 0; i < N; i++, c++) {
        if (10 * c >= N) {
            printf("Progress: %u/%lu\n", i + 1, N);
            c = 0;
        }

        auto [u, mu_e] = plasmon_density_ts<T, green_func, 0, det_zero_f>(
            n_vec[i - 2],
            i > 2 ? u_vec[i - 1] : std::numeric_limits<double>::quiet_NaN(),
            s);

        u_vec[i]  = u;
        result[i] = mu_e;
    }
    printf("Progress: %lu/%lu\n", N, N);

    return result;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    uint32_t N,
    double (*det_zero)(
        double,
        double,
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>&,
        double)>
double plasmon_density_exc_t(
    double n_exc,
    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, N>>>& s) {
    /*
     * Compute first u from n_exc, so that we can then compute mu_e
     * and finally n_e.
     */

    double u{s.mat_s.sys.mu_exc_u(n_exc)};
    double mu_e;

    if (u > -10) {
        mu_e = plasmon_exc_mu_lim_t<T, green_func, N, det_zero>(
            s.mat_s, logExp(u) / s.mat_s.sys.beta, s.mu_e_lim, s.eb_lim);
    } else {
        mu_e = s.mu_e_lim;
    }

    return mu_e;
}

std::vector<double> plasmon_density_exc_ht_v(
    const std::vector<double>& n_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta) {
    using T                   = double;
    constexpr auto green_func = plasmon_green_ht<T>;
    constexpr auto det_zero   = plasmon_det_zero_t<T, green_func, true>;
    constexpr auto det_zero_f = plasmon_det_zero_t<T, green_func, false>;

    plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>> mat_s(
        N_k, 1, sys, delta);

    double mu_e_lim{plasmon_exc_mu_lim_t<T, green_func, 0, det_zero>(mat_s)};
    double eb_lim{
        det_zero(
            mu_e_lim, sys.get_mu_h(mu_e_lim), mat_s,
            std::numeric_limits<double>::quiet_NaN()),
    };

    printf("[%s] mu_e: %f, eb: %f\n", __func__, mu_e_lim, eb_lim);

    plasmon_density_s<
        plasmon_mat_s<T, plasmon_potcoef<T, green_func, false, 0>>>
        s{
            mat_s,
            mu_e_lim,
            eb_lim,
        };

    uint64_t N{n_vec.size() + 2};
    std::vector<double> result(N);

    result[0] = mu_e_lim;
    result[1] = eb_lim;

    printf("Progress: %u/%lu\n", 0, N);
    for (uint32_t i = 2, c = 0; i < N; i++, c++) {
        if (10 * c >= N) {
            printf("Progress: %u/%lu\n", i + 1, N);
            c = 0;
        }
        result[i] = plasmon_density_exc_t<T, green_func, 0, det_zero_f>(
            n_vec[i - 2], s);
    }
    printf("Progress: %lu/%lu\n", N, N);

    return result;
}

std::vector<std::complex<double>> plasmon_cond_v(
    const std::vector<double>& q_yield_vec,
    const std::vector<double>& Na_vec,
    double L,
    double mob_R,
    double mob_I,
    double pol,
    double freq,
    const system_data& sys) {
    uint64_t N{q_yield_vec.size()};
    std::vector<std::complex<double>> result(N);

#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++) {
        result[i] = std::complex<double>(
            Na_vec[i] * sys.c_e_charge / L * mob_R * q_yield_vec[i],
            Na_vec[i] * sys.c_e_charge / L * mob_I * q_yield_vec[i] +
                2 * M_PI * pol * freq * Na_vec[i] / L *
                    (1.0 - q_yield_vec[i]));
    }

    return result;
}
