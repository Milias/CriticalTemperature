#include "plasmons.h"
#include "plasmons_utils.h"

template <
    uint8_t remove_term = 0,
    bool invert         = true,
    typename T          = std::complex<double>,
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

template <
    bool invert      = true,
    typename T       = std::complex<double>,
    bool include_cou = true>
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

    std::vector<std::complex<double>> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] = plasmon_green(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
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

    std::vector<std::complex<double>> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] =
            plasmon_green<0, false>(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
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

    std::vector<std::complex<double>> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] = plasmon_green_ht(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
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

    std::vector<std::complex<double>> result(N_total);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_total; i++) {
        std::vector<double> t_v{wk_vec[i]};
        result[i] =
            plasmon_green_ht<false>(t_v[0], t_v[1], mu_e, mu_h, sys, delta);
    }

    return result;
}

double plasmon_kmax_f(double k, void* params) {
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    double w{s->sys.m_pe * k * k + 2.0 * std::sqrt(s->sys.m_pe * s->mu_e) * k};

    return plasmon_green<1, false>(w, k, s->mu_e, s->mu_h, s->sys).real();
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

    struct plasmon_potcoef_s s {
        0, 0, 0, mu_e, mu_h, sys
    };

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
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    return plasmon_green<0, false>(w, s->k1, s->mu_e, s->mu_h, s->sys).real();
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

    struct plasmon_potcoef_s s {
        0, k, 0, mu_e, mu_h, sys
    };

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
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    std::complex<double> result{
        plasmon_green<0, false>(s->w, k, s->mu_e, s->mu_h, s->sys)};

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

    struct plasmon_potcoef_s s {
        w, 0, 0, mu_e, mu_h, sys
    };

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

template <std::complex<double> (*green_func)(
    double, double, double, double, const system_data& sys, double)>
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
    const std::vector<double> wkk,
    double mu_e,
    double mu_h,
    const system_data& sys) {
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

    struct plasmon_potcoef_s s {
        w, wkk[1], wkk[2], mu_e, mu_h, sys
    };

    gsl_function funct;
    funct.function = &plasmon_disp_th_f<plasmon_green<0, false>>;
    funct.params   = &s;

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
        double, double, double, double, const system_data& sys, double)>
auto plasmon_potcoef_f(double th, void* params) {
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    double k{
        std::sqrt(
            s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)),
    };

    T green{
        green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta),
    };

    if constexpr (return_part == 0 || std::is_same<T, double>::value) {
        return green;
    } else if constexpr (return_part == 1) {
        return green.real();
    } else if constexpr (return_part == 2) {
        return green.imag();
    }
}

template <
    typename T = std::complex<double>,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool find_pole = true>
T plasmon_potcoef(
    const std::vector<double>& wkk,
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

    integrands[0].function = &plasmon_potcoef_f<T, 1, green_func>;
    integrands[0].params   = &s;

    constexpr uint32_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-8};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    if constexpr (find_pole && std::is_same<T, std::complex<double>>::value) {
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
        gsl_integration_qag(
            integrands, M_PI, 0, local_eps, 0, local_ws_size,
            GSL_INTEG_GAUSS21, ws, result, error);
    }

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        integrands[1].function = &plasmon_potcoef_f<T, 2, green_func>;
        integrands[1].params   = &s;

        gsl_integration_qag(
            integrands + 1, 0, M_PI, global_eps, 0, local_ws_size,
            GSL_INTEG_GAUSS31, ws, result + 1, error + 1);
    }

    gsl_integration_workspace_free(ws);

    /*
    for (uint32_t i = 0; i < n_int; i++) {
        printf("%s [%d]: %f (%e)\n", __func__, i, result[i], error[i]);
    }
    printf("\n");
    */

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        return {result[0] * M_1_PI, result[1] * M_1_PI};
    } else {
        return result[0] * M_1_PI;
    }
}

template <
    typename T,
    T (*potcoef_func)(
        const std::vector<double>& wkk,
        double mu_e,
        double mu_h,
        const system_data& sys,
        double delta)>
arma::Mat<T> plasmon_potcoef_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    // printf("%s started\n", __func__);
    arma::vec u0{arma::linspace(1.0 / N_k, 1.0 - 1.0 / N_k, N_k)};

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        arma::Cube<T> potcoef(N_k, N_k, N_w);

        arma::vec u1 = arma::linspace(0, 1.0 - 1.0 / N_w, N_w);

#pragma omp parallel for collapse(3)
        for (uint32_t i = 0; i < N_k; i++) {
            for (uint32_t j = 0; j < N_k; j++) {
                for (uint32_t k = 0; k < N_w; k++) {
                    const std::vector<double> wkk{
                        u1(k) / (1.0 - std::pow(u1(k), 2)),
                        (1.0 - u0(i)) / u0(i),
                        (1.0 - u0(j)) / u0(j),
                    };

                    potcoef(i, j, k) =
                        potcoef_func(wkk, mu_e, mu_h, sys, delta);
                }
            }
        }

        if (N_w > 1) {
            arma::Mat<T> result(N_k * N_w, N_k * N_w);

            for (uint32_t i = 0; i < N_w; i++) {
                for (uint32_t j = 0; j < i; j++) {
                    result.submat(
                        N_k * i, N_k * j, N_k * (i + 1) - 1,
                        N_k * (j + 1) - 1) = potcoef.slice(i - j);
                }

                for (uint32_t j = i; j < N_w; j++) {
                    result.submat(
                        N_k * i, N_k * j, N_k * (i + 1) - 1,
                        N_k * (j + 1) - 1) = arma::conj(potcoef.slice(j - i));
                }
            }
            return result;

        } else {
            return potcoef.slice(0);
        }

    } else {
        arma::Mat<T> potcoef(N_k, N_k);

#pragma omp parallel for
        for (uint32_t i = 0; i < N_k; i++) {
            for (uint32_t j = 0; j <= i; j++) {
                const std::vector<double> wkk{
                    0.0,
                    (1.0 - u0(i)) / u0(i),
                    (1.0 - u0(j)) / u0(j),
                };

                double r = potcoef_func(wkk, mu_e, mu_h, sys, delta);

                potcoef(i, j) = r;
                potcoef(j, i) = r;
            }
        }

        return potcoef;
    }
}

std::vector<std::complex<double>> plasmon_potcoef_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    arma::cx_mat result{
        plasmon_potcoef_mat<
            std::complex<double>,
            plasmon_potcoef<std::complex<double>, plasmon_green>>(
            N_k, N_w, mu_e, mu_h, sys, delta),
    };

    std::vector<std::complex<double>> result_vec(result.n_elem);

#pragma omp parallel for
    for (uint32_t i = 0; i < result.n_elem; i++) {
        result_vec[i] = result(i);
    }

    return result_vec;
}

std::vector<std::complex<double>> plasmon_potcoef_ht_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    arma::cx_mat result{
        plasmon_potcoef_mat<
            std::complex<double>,
            plasmon_potcoef<std::complex<double>, plasmon_green_ht, false>>(
            N_k, N_w, mu_e, mu_h, sys, delta),
    };

    std::vector<std::complex<double>> result_vec(result.n_elem);

#pragma omp parallel for
    for (uint32_t i = 0; i < result.n_elem; i++) {
        result_vec[i] = result(i);
    }

    return result_vec;
}

template <uint8_t return_part = 0, typename T, typename T2>
auto plasmon_det_f(T2 z, void* params) {
    plasmon_mat_s<T>* s{static_cast<plasmon_mat_s<T>*>(params)};

    s->mat_z_G0.fill(-z);
    s->mat_z_G0 += s->mat_G0;
    s->mat_z_G0 = 1.0 / s->mat_z_G0;

    s->mat_potcoef.fill(arma::fill::eye);
    s->mat_potcoef += s->mat_elem.each_row() % s->mat_z_G0;

    if constexpr (return_part == 0) {
        return arma::det(s->mat_potcoef);
    } else if constexpr (return_part == 1) {
        return arma::det(s->mat_potcoef).real();
    } else if constexpr (return_part == 2) {
        return arma::det(s->mat_potcoef).imag();
    } else if constexpr (return_part == 3) {
        return s->mat_potcoef;
    }
}

template <typename T>
void plasmon_fill_mat_G0(
    uint32_t N_k, uint32_t N_w, const system_data& sys, arma::Row<T>& mat_G0) {
    arma::vec u0{arma::linspace(1.0 / N_k, 1.0 - 1.0 / N_k, N_k)};
    arma::vec k_v{(1.0 - u0) / u0};

    arma::vec diag_vals{
        -0.5 * std::pow(sys.c_hbarc, 2) * arma::pow(k_v, 2) / sys.m_p,
    };

    arma::Row<T> diag_vals_t(N_k);

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        diag_vals_t = arma::Row<T>(diag_vals.t().eval(), arma::zeros(N_k));
    } else {
        diag_vals_t = diag_vals.t().eval();
    }

    for (uint32_t i = 0; i < N_w; i++) {
        mat_G0.cols(i * N_k, (i + 1) * N_k - 1) = diag_vals_t;
    }
}

template <typename T = std::complex<double>>
void plasmon_fill_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta,
    const arma::Mat<T>& potcoef,
    arma::Mat<T>& mat_elem) {
    // printf("%s started\n", __func__);
    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    arma::vec u0{arma::linspace(1.0 / N_k, 1.0 - 1.0 / N_k, N_k)};
    arma::vec u1(N_w);
    double du0{u0(1) - u0(0)}, du1;

    if (N_w > 1) {
        u1  = arma::linspace(0, 1.0 - 1.0 / N_w, N_w);
        du1 = u1(1) - u0(0);

#pragma omp parallel for collapse(4)
        for (uint32_t i = 0; i < N_w; i++) {
            for (uint32_t j = 0; j < N_w; j++) {
                for (uint32_t k = 0; k < N_k; k++) {
                    for (uint32_t l = 0; l < N_k; l++) {
                        const std::vector<double> k_v{
                            (1.0 - u0(k)) / u0(k),
                            (1.0 - u0(l)) / u0(l),
                        };

                        mat_elem(i + N_w * k, j + N_w * l) =
                            du0 * du1 * k_v[1] * (1 + std::pow(u1(j), 2)) /
                            std::pow(u0(l) * (1 - std::pow(u1(j), 2)), 2) *
                            potcoef(i + N_w * k, j + N_w * l);
                    }
                }
            }
        }
    } else {
#pragma omp parallel for collapse(2)
        for (uint32_t i = 0; i < N_k; i++) {
            for (uint32_t k = 0; k < N_k; k++) {
                const double t_v[2] = {u0(i), u0(k)};

                const std::vector<double> k_v{
                    (1.0 - t_v[0]) / t_v[0],
                    (1.0 - t_v[1]) / t_v[1],
                };

                mat_elem(i, k) =
                    du0 * k_v[1] / std::pow(t_v[1], 2) * potcoef(i, k);
            }
        }
    }
}

template <
    typename T = std::complex<double>,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool return_matrix  = false,
    bool print_progress = false>
auto plasmon_det_t(
    const std::vector<T>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    uint64_t N_z{z_vec.size()};

    // Constant
    arma::Mat<T> mat_elem(N_k * N_w, N_k * N_w);

    arma::Row<T> mat_G0(N_k * N_w);
    plasmon_fill_mat_G0<T>(N_k, N_w, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k * N_w);
    arma::Mat<T> mat_potcoef{
        plasmon_potcoef_mat<T, plasmon_potcoef<T, green_func>>(
            N_k, N_w, mu_e, mu_h, sys, delta),
    };

    if constexpr (print_progress) {
        printf("Initializing\n");
    }

    plasmon_fill_mat<T>(
        N_k, N_w, mu_e, mu_h, sys, delta, mat_potcoef, mat_elem);

    struct plasmon_mat_s<T> s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef
    };

    if constexpr (return_matrix) {
        std::vector<std::vector<T>> result(N_z);
        arma::Mat<T> mat_result(N_k * N_w, N_k * N_w);

        if constexpr (print_progress) {
            printf("Starting\n");
            printf("Progress: %u/%lu\n", 0, N_z);
        }
        for (uint32_t i = 0, c = 0; i < N_z; i++, c++) {
            if constexpr (print_progress) {
                if (10 * c >= N_z) {
                    printf("Progress: %u/%lu\n", i + 1, N_z);
                    c = 0;
                }
            }

            mat_result = plasmon_det_f<3, T>(z_vec[i], &s);

            result[i].resize(mat_elem.n_elem);

#pragma omp parallel for
            for (uint32_t j = 0; j < mat_elem.n_elem; j++) {
                result[i][j] = mat_result(j);
            }
        }
        if constexpr (print_progress) {
            printf("Progress: %lu/%lu\n", N_z, N_z);
        }

        return result;

    } else {
        std::vector<T> result(N_z);

        if constexpr (print_progress) {
            printf("Starting\n");
            printf("Progress: %u/%lu\n", 0, N_z);
        }
        for (uint32_t i = 0, c = 0; i < N_z; i++, c++) {
            if constexpr (print_progress) {
                if (10 * c >= N_z) {
                    printf("Progress: %u/%lu\n", i + 1, N_z);
                    c = 0;
                }
            }

            result[i] = plasmon_det_f<0, T>(z_vec[i], &s);
        }

        if constexpr (print_progress) {
            printf("Progress: %lu/%lu\n", N_z, N_z);
        }

        return result;
    }
}

std::vector<std::complex<double>> plasmon_fmat_cx(
    const std::complex<double>& z,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_t<std::complex<double>, plasmon_green, true, true>(
        {z}, N_k, N_w, mu_e, mu_h, sys, delta)[0];
}

std::vector<std::complex<double>> plasmon_fmat_ht_cx(
    const std::complex<double>& z,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_t<std::complex<double>, plasmon_green_ht, true, true>(
        {z}, N_k, N_w, mu_e, mu_h, sys, delta)[0];
}

std::vector<double> plasmon_det(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_t<double, plasmon_green, false, true>(
        z_vec, N_k, N_w, mu_e, mu_h, sys, delta);
}

std::vector<double> plasmon_det_ht(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_t<double, plasmon_green_ht, false, true>(
        z_vec, N_k, N_w, mu_e, mu_h, sys, delta);
}

std::vector<std::complex<double>> plasmon_det_cx(
    const std::vector<std::complex<double>>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_t<std::complex<double>, plasmon_green, false, true>(
        z_vec, N_k, N_w, mu_e, mu_h, sys, delta);
}

std::vector<std::complex<double>> plasmon_det_ht_cx(
    const std::vector<std::complex<double>>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_t<std::complex<double>, plasmon_green_ht, false, true>(
        z_vec, N_k, N_w, mu_e, mu_h, sys, delta);
}

template <
    typename T = std::complex<double>,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool sweep = true>
double plasmon_det_zero_t(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    plasmon_mat_s<T>& s,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta  = 1e-12) {
    constexpr double local_eps{1e-8};

    s.mat_potcoef = plasmon_potcoef_mat<T, plasmon_potcoef<T, green_func>>(
        N_k, N_w, mu_e, mu_h, sys, delta);

    plasmon_fill_mat<T>(
        N_k, N_w, mu_e, mu_h, sys, delta, s.mat_potcoef, s.mat_elem);

    gsl_function funct;
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        funct.function = &plasmon_det_f<1, T>;
    } else {
        funct.function = &plasmon_det_f<0, T>;
    }
    funct.params = &s;

    double z{-sys.get_E_n(0.5)};
    double z_min{std::isnan(eb_min) ? 0.75 * z : -eb_min}, z_max{z};

    if (std::abs(funct.function(z_max, &s)) < local_eps) {
        return -z_max;
    }

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
    double delta) {
    using T = double;

    // Constant
    arma::Mat<T> mat_elem(N_k, N_k);
    arma::Row<T> mat_G0(N_k);
    plasmon_fill_mat_G0<T>(N_k, 1, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    return plasmon_det_zero_t<T, plasmon_green>(
        N_k, 1, mu_e, mu_h, sys, mat_s,
        std::numeric_limits<double>::quiet_NaN(), 0);
}

double plasmon_det_zero_ht(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    using T = double;

    // Constant
    arma::Mat<T> mat_elem(N_k, N_k);
    arma::Row<T> mat_G0(N_k);
    plasmon_fill_mat_G0<T>(N_k, 1, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    return plasmon_det_zero_t<T, plasmon_green_ht>(
        N_k, 1, mu_e, mu_h, sys, mat_s,
        std::numeric_limits<double>::quiet_NaN(), 0);
}

double plasmon_det_zero_lwl(uint32_t N_k, double ls, const system_data& sys) {
    using T = double;

    // Constant
    arma::Mat<T> mat_elem(N_k, N_k);
    arma::Row<T> mat_G0(N_k);
    plasmon_fill_mat_G0<T>(N_k, 1, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    return plasmon_det_zero_t<T, plasmon_green_lwl>(
        N_k, 1, ls, 0, sys, mat_s, std::numeric_limits<double>::quiet_NaN(),
        0);
}

double plasmon_det_zero_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    using T = std::complex<double>;

    // Constant
    arma::Mat<T> mat_elem(N_k * N_w, N_k * N_w);
    arma::Row<T> mat_G0(N_k * N_w);
    plasmon_fill_mat_G0<T>(N_k, N_w, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k * N_w);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    return plasmon_det_zero_t<T, plasmon_green>(
        N_k, N_w, mu_e, mu_h, sys, mat_s,
        std::numeric_limits<double>::quiet_NaN(), delta);
}

double plasmon_det_zero_ht_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    using T = std::complex<double>;

    // Constant
    arma::Mat<T> mat_elem(N_k * N_w, N_k * N_w);
    arma::Row<T> mat_G0(N_k * N_w);
    plasmon_fill_mat_G0<T>(N_k, N_w, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k * N_w);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    return plasmon_det_zero_t<T, plasmon_green_ht>(
        N_k, N_w, mu_e, mu_h, sys, mat_s,
        std::numeric_limits<double>::quiet_NaN(), delta);
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
    struct plasmon_exc_mu_zero_s s {
        sys
    };

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

    double result;

    result = x + s->sys.get_mu_h(x) - s->val;

    return result;
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

    struct plasmon_exc_mu_zero_s s {
        sys, val
    };

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

    if (std::abs(funct.f(z, &s)) < 1e-14) {
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
        status = gsl_root_test_residual(funct.f(z, &s), 1e-10);
    }

    gsl_root_fdfsolver_free(solver);

    return z;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    double (*det_zero)(
        uint32_t,
        uint32_t,
        double,
        double,
        const system_data&,
        plasmon_mat_s<T>&,
        double,
        double)>
double plasmon_exc_mu_lim_f(double mu_e, void* params) {
    plasmon_exc_mu_lim_s<T>* s{static_cast<plasmon_exc_mu_lim_s<T>*>(params)};

    double mu_h{s->sys.get_mu_h(mu_e)};

    return mu_e + mu_h -
           det_zero(
               s->N_k, s->N_w, mu_e, mu_h, s->sys, s->mat_s,
               s->eb_lim + s->val, s->delta) +
           s->val;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    double (*det_zero)(
        uint32_t,
        uint32_t,
        double,
        double,
        const system_data&,
        plasmon_mat_s<T>&,
        double,
        double)>
double plasmon_exc_mu_lim_t(
    uint32_t N_k,
    uint32_t N_w,
    const system_data& sys,
    double val    = 0.0,
    double mu_lim = std::numeric_limits<double>::quiet_NaN(),
    double eb_lim = std::numeric_limits<double>::quiet_NaN(),
    double delta  = 1e-12) {
    // Constant
    arma::Mat<T> mat_elem(N_k * N_w, N_k * N_w);
    arma::Row<T> mat_G0(N_k * N_w);
    plasmon_fill_mat_G0<T>(N_k, N_w, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k * N_w);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    struct plasmon_exc_mu_lim_s<T> s {
        N_k, N_w, sys, mat_s, val, eb_lim, delta
    };

    gsl_function funct;
    funct.function = &plasmon_exc_mu_lim_f<T, green_func, det_zero>;
    funct.params   = &s;

    double z{0};
    double z_min{
        sys.get_E_n(0.5) - std::log(sys.m_eh) / sys.beta - val,
    };
    double z_max{
        val == 0 || std::isnan(mu_lim) ? plasmon_exc_mu_zero(sys) : mu_lim,
    };

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
    return z;
}

double plasmon_exc_mu_lim_ht(
    uint32_t N_k, const system_data& sys, double val) {
    return plasmon_exc_mu_lim_t<
        double, plasmon_green_ht,
        plasmon_det_zero_t<double, plasmon_green_ht, true>>(
        N_k, 1, sys, val, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(), 0.0);
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    double (*det_zero)(
        uint32_t,
        uint32_t,
        double,
        double,
        const system_data&,
        plasmon_mat_s<T>&,
        double,
        double)>
void plasmon_mu_e_u(
    double u,
    plasmon_density_s<T>* s,
    double& mu_e,
    double& mu_h,
    double& eb) {
    if (u > -10) {
        mu_e = plasmon_exc_mu_lim_t<T, green_func, det_zero>(
            s->N_k, s->N_w, s->sys, logExp(u) / s->sys.beta, s->mu_e_lim,
            s->eb_lim, s->delta);

        mu_h = s->sys.get_mu_h(mu_e);

        eb = det_zero(
            s->N_k, s->N_w, mu_e, mu_h, s->sys, s->mat_s, s->eb_lim, s->delta);
    } else if (u > 10) {
        mu_e = plasmon_exc_mu_val(
            -s->sys.get_E_n(0.5) + logExp(u) / s->sys.beta, s->sys);
        mu_h = s->sys.get_mu_h(mu_e);
        eb   = s->sys.get_E_n(0.5);
    } else {
        mu_e = s->mu_e_lim;
        mu_h = s->sys.get_mu_h(mu_e);
        eb   = s->eb_lim;
    }
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    double (*det_zero)(
        uint32_t,
        uint32_t,
        double,
        double,
        const system_data&,
        plasmon_mat_s<T>&,
        double,
        double),
    bool substract_total>
double plasmon_density_mu_f(double u, void* params) {
    /*
     * eb - mu_ex = mu_ex * log(1 + exp(u))
     */
    plasmon_density_s<T>* s{static_cast<plasmon_density_s<T>*>(params)};

    double mu_e, mu_h, eb;
    plasmon_mu_e_u<T, green_func, det_zero>(u, s, mu_e, mu_h, eb);

    /*
    printf(
        "[%s] mu_e_lim: %f, eb_lim: %f, u: %f, %.16f, %.16f\n", __func__,
        s->mu_e_lim, s->eb_lim, u, mu_e, eb);
    */

    if constexpr (substract_total) {
        return 2 * s->sys.density_ideal(mu_e) + s->sys.density_exc_exp(u, eb) -
               s->n_total;
    } else {
        return 2 * s->sys.density_ideal(mu_e) + s->sys.density_exc_exp(u, eb);
    }
}

std::vector<double> plasmon_density_mu_ht_v(
    const std::vector<double>& u_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta) {
    using T = double;

    // Constant
    arma::Mat<T> mat_elem(N_k, N_k);
    arma::Row<T> mat_G0(N_k);
    plasmon_fill_mat_G0<T>(N_k, 1, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    double mu_e_lim{plasmon_exc_mu_lim_ht(N_k, sys)};
    double eb_lim{
        plasmon_det_zero_t<T, plasmon_green_ht, true>(
            N_k, 1, mu_e_lim, sys.get_mu_h(mu_e_lim), sys, mat_s,
            std::numeric_limits<double>::quiet_NaN(), delta),
    };

    // printf("[%s] mu_e: %f, eb: %f\n", __func__, mu_e_lim, eb_lim);

    struct plasmon_density_s<T> s {
        0.0, N_k, 1, mu_e_lim, eb_lim, sys, mat_s, delta
    };

    uint64_t N_z{u_vec.size()};

    std::vector<double> result(N_z);

    printf("Progress: %u/%lu\n", 0, N_z);
    for (uint32_t i = 0, c = 0; i < N_z; i++, c++) {
        if (10 * c >= N_z) {
            printf("Progress: %u/%lu\n", i + 1, N_z);
            c = 0;
        }

        result[i] = plasmon_density_mu_f<
            T, plasmon_green_ht,
            plasmon_det_zero_t<T, plasmon_green_ht, false>, false>(
            u_vec[i], (void*)&s);
    }
    printf("Progress: %lu/%lu\n", N_z, N_z);

    return result;
}

template <
    typename T = double,
    T (*green_func)(
        double, double, double, double, const system_data& sys, double),
    double (*det_zero)(
        uint32_t,
        uint32_t,
        double,
        double,
        const system_data&,
        plasmon_mat_s<T>&,
        double,
        double)>
double plasmon_density_t(
    double n_total,
    uint32_t N_k,
    const system_data& sys,
    plasmon_density_s<T>& s,
    double delta = 1e-12) {
    s.n_total = n_total;

    gsl_function funct;
    funct.function = &plasmon_density_mu_f<T, green_func, det_zero, true>;
    funct.params   = &s;

    /*
    double mu_e{sys.mu_ideal(n_total)};
    double mu_h{sys.get_mu_h(mu_e)};
    double eb{sys.get_E_n(0.5)};
    */

    /*
     * Seems like f(z_max) is always negative.
     */

    double z{0};
    double z_max{sys.mu_exc_u(n_total)};
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
        z_min = 2 * z_max;
    }

    double f_min;
    bool return_nan{true};
    uint32_t max_sweep_iter{10};
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

    if (return_nan) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double f_max{funct.function(z_max, &s)};

    printf(
        "[%s] min: %.16f, max: %.16f, f: (%.3e, %.3e)\n", __func__, z_min,
        z_max, f_min, f_max);

    const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z      = gsl_root_fsolver_root(solver);
        z_min  = gsl_root_fsolver_x_lower(solver);
        z_max  = gsl_root_fsolver_x_upper(solver);

        printf("[%s] min: %.16f, max: %.16f\n", __func__, z_min, z_max);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

std::vector<double> plasmon_density_ht_v(
    const std::vector<double>& n_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta) {
    using T = double;

    // Constant
    arma::Mat<T> mat_elem(N_k, N_k);
    arma::Row<T> mat_G0(N_k);
    plasmon_fill_mat_G0<T>(N_k, 1, sys, mat_G0);

    // Change every step
    arma::Row<T> mat_z_G0(N_k);
    arma::Mat<T> mat_potcoef;

    struct plasmon_mat_s<T> mat_s {
        mat_elem, mat_G0, mat_z_G0, mat_potcoef,
    };

    double mu_e_lim{plasmon_exc_mu_lim_ht(N_k, sys)};
    double eb_lim{
        plasmon_det_zero_t<T, plasmon_green_ht, true>(
            N_k, 1, mu_e_lim, sys.get_mu_h(mu_e_lim), sys, mat_s,
            std::numeric_limits<double>::quiet_NaN(), delta),
    };

    printf("[%s] mu_e: %f, eb: %f\n", __func__, mu_e_lim, eb_lim);

    struct plasmon_density_s<T> s {
        0.0, N_k, 1, mu_e_lim, eb_lim, sys, mat_s, delta
    };

    uint64_t N{n_vec.size()};
    std::vector<double> result(N);

    for (uint32_t i = 0; i < N; i++) {
        result[i] = plasmon_density_t<
            T, plasmon_green_ht,
            plasmon_det_zero_t<T, plasmon_green_ht, false>>(
            n_vec[i], N_k, sys, s, delta);
    }

    return result;
}
