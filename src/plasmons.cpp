#include "plasmons.h"
#include "plasmons_utils.h"

template <uint8_t remove_term = 0, bool invert = true>
std::complex<double> plasmon_green(
    double w,
    double k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12) {
    const std::complex<double> w_complex(w, delta);

    const double k2{k * k};

    std::complex<double> pi_screen_nofactor[2];

    double E[2] = {sys.m_pe * sys.c_alpha * k2, sys.m_ph * sys.c_alpha * k2};

    std::complex<double> nu[4] = {
        -w_complex / E[0] - 1.0, -w_complex / E[0] + 1.0,
        -w_complex / E[1] - 1.0, -w_complex / E[1] + 1.0};

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

    std::complex<double> result{
        -sys.eps_r * k / (sys.c_hbarc * sys.c_aEM) +
            0.125 * M_1_PI / sys.c_alpha *
                (pi_screen_nofactor[0] / sys.m_pe +
                 pi_screen_nofactor[1] / sys.m_ph),
    };

    if constexpr (invert) {
        return 1.0 / result;
    } else {
        return result;
    }
}

template <bool invert = true>
std::complex<double> plasmon_green_ht(
    double w,
    double k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12) {
    const std::complex<double> w_complex(w, delta);

    const double k2{k * k};

    std::complex<double> pi_screen_nofactor[2];

    const double E[2] = {sys.m_pe * sys.c_alpha * k2,
                         sys.m_ph * sys.c_alpha * k2};

    std::complex<double> nu[4] = {
        -w_complex - E[0],
        -w_complex + E[0],
        -w_complex - E[1],
        -w_complex + E[1],
    };

    const std::complex<double> nu_exp[2] = {
        mu_e - 0.25 * (std::pow(w_complex, 2) + std::pow(E[0], 2)) / E[0],
        mu_h - 0.25 * (std::pow(w_complex, 2) + std::pow(E[1], 2)) / E[1],
    };

    pi_screen_nofactor[0] =
        std::exp(sys.beta * nu_exp[0]) / std::sqrt(sys.beta * E[0]) *
        (std::complex<double>(0, 2.0) * std::sinh(0.5 * sys.beta * w_complex) -
         std::exp(-0.5 * sys.beta * w_complex) *
             erf_cx(std::sqrt(0.25 * sys.beta / E[0]) * nu[0]) +
         std::exp(0.5 * sys.beta * w_complex) *
             erf_cx(std::sqrt(0.25 * sys.beta / E[0]) * nu[1]));

    pi_screen_nofactor[0] =
        std::exp(sys.beta * nu_exp[1]) / std::sqrt(sys.beta * E[1]) *
        (std::complex<double>(0, 2.0) * std::sinh(0.5 * sys.beta * w_complex) -
         std::exp(-0.5 * sys.beta * w_complex) *
             erf_cx(std::sqrt(0.25 * sys.beta / E[1]) * nu[2]) +
         std::exp(0.5 * sys.beta * w_complex) *
             erf_cx(std::sqrt(0.25 * sys.beta / E[1]) * nu[3]));

    const std::complex<double> result{
        -sys.eps_r * k / (sys.c_hbarc * sys.c_aEM) -
            0.125 / (sys.c_alpha * M_SQRTPI) *
                (pi_screen_nofactor[0] / sys.m_pe +
                 pi_screen_nofactor[1] / sys.m_ph),
    };

    if constexpr (invert) {
        return 1.0 / result;
    } else {
        return result;
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

double plasmon_potcoef_lwl_f(double th, void* params) {
    plasmon_potcoef_lwl_s* s{static_cast<plasmon_potcoef_lwl_s*>(params)};

    const double k2{s->k1 * s->k1 + s->k2 * s->k2 -
                    2 * s->k1 * s->k2 * std::cos(th)};

    return -1.0 / (s->sys.eps_r * (std::sqrt(k2) + s->ls) /
                   (s->sys.c_hbarc * s->sys.c_aEM));
}

double plasmon_kmax_f(double k, void* params) {
    plasmon_kmax_s* s{static_cast<plasmon_kmax_s*>(params)};

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

    struct plasmon_kmax_s s {
        mu_e, mu_h, sys
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
    plasmon_disp_s* s{static_cast<plasmon_disp_s*>(params)};

    return plasmon_green<0, false>(w, s->k, s->mu_e, s->mu_h, s->sys).real();
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

    struct plasmon_disp_s s {
        k, mu_e, mu_h, sys
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
    plasmon_disp_inv_s* s{static_cast<plasmon_disp_inv_s*>(params)};
    std::complex<double> result{
        plasmon_green<0, false>(s->w, k, s->mu_e, s->mu_h, s->sys)};
    return result.real() + result.imag();
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

    struct plasmon_disp_inv_s s {
        w, mu_e, mu_h, sys
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
    plasmon_disp_th_s* s{static_cast<plasmon_disp_th_s*>(params)};

    double k{std::sqrt(
        s->k0 * s->k0 + s->k1 * s->k1 - 2 * s->k0 * s->k1 * std::cos(th))};

    std::complex<double> result{
        green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta)};

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

    double k2_max{2 * sys.m_e * mu_e *
                  std::pow((sqrt(1 + w / mu_e) - 1) / sys.c_hbarc, 2)};

    if (k2_max > std::pow(wkk[1] + wkk[2], 2) ||
        k2_max < std::pow(wkk[1] - wkk[2], 2)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double th_max{std::acos(
        (k2_max - wkk[1] * wkk[1] - wkk[2] * wkk[2]) /
        (-2 * wkk[1] * wkk[2]))};

    double z, z_min{th_max * 1e-5}, z_max{th_max * (1 - 1e-5)};

    struct plasmon_disp_th_s s {
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
    uint8_t return_part,
    std::complex<double> (*green_func)(
        double, double, double, double, const system_data& sys, double)>
auto plasmon_potcoef_f(double th, void* params) {
    plasmon_potcoef_s* s{static_cast<plasmon_potcoef_s*>(params)};

    double k{std::sqrt(
        s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th))};

    std::complex<double> green{
        green_func(s->w, k, s->mu_e, s->mu_h, s->sys, s->delta)};

    if constexpr (return_part == 0) {
        return green;
    } else if constexpr (return_part == 1) {
        return green.real();
    } else if constexpr (return_part == 2) {
        return green.imag();
    }
}

template <
    std::complex<double> (*green_func)(
        double, double, double, double, const system_data& sys, double),
    typename T     = std::complex<double>,
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

    integrands[0].function = &plasmon_potcoef_f<1, green_func>;
    integrands[0].params   = &s;

    constexpr uint32_t local_ws_size{(1 << 10)};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    if constexpr (find_pole) {
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
            integrands, M_PI, 0, global_eps, 0, local_ws_size,
            GSL_INTEG_GAUSS31, ws, result, error);
    }

    if constexpr (std::is_same<T, std::complex<double>>::value) {
        integrands[1].function = &plasmon_potcoef_f<2, green_func>;
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
    auto (*potcoef_func)(
        const std::vector<double>& wkk,
        double mu_e,
        double mu_h,
        const system_data& sys,
        double delta),
    typename T = std::complex<double>>
arma::Mat<T> plasmon_potcoef_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    arma::Cube<T> potcoef(N_k, N_k, N_w);

    arma::vec u0{arma::linspace(1.0 / N_k, 1.0 - 1.0 / N_k, N_k)};
    arma::vec u1(N_w);

    if (N_w > 1) {
        u1 = arma::linspace(0, 1.0 - 1.0 / N_w, N_w);
    } else {
        u1 = {0.0};
    }

#pragma omp parallel for collapse(3)
    for (uint32_t i = 0; i < N_k; i++) {
        for (uint32_t j = 0; j < N_k; j++) {
            for (uint32_t k = 0; k < N_w; k++) {
                const std::vector<double> wkk{
                    u1(k) / (1.0 - std::pow(u1(k), 2)),
                    (1.0 - u0(i)) / u0(i),
                    (1.0 - u0(j)) / u0(j),
                };

                potcoef(i, j, k) = potcoef_func(wkk, mu_e, mu_h, sys, delta);
            }
        }
    }

    arma::Mat<T> result(N_k * N_w, N_k * N_w);

    for (uint32_t i = 0; i < N_w; i++) {
        for (uint32_t j = 0; j < i; j++) {
            result.submat(
                N_k * i, N_k * j, N_k * (i + 1) - 1, N_k * (j + 1) - 1) =
                potcoef.slice(i - j);
        }

        for (uint32_t j = i; j < N_w; j++) {
            result.submat(
                N_k * i, N_k * j, N_k * (i + 1) - 1, N_k * (j + 1) - 1) =
                arma::conj(potcoef.slice(j - i));
        }
    }

    return result;
}

std::vector<std::complex<double>> plasmon_potcoef_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    arma::cx_mat result{plasmon_potcoef_mat<plasmon_potcoef<plasmon_green>>(
        N_k, N_w, mu_e, mu_h, sys, delta)};

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
    arma::cx_mat result{plasmon_potcoef_mat<
        plasmon_potcoef<plasmon_green_ht, std::complex<double>, false>>(
        N_k, N_w, mu_e, mu_h, sys, delta)};

    std::vector<std::complex<double>> result_vec(result.n_elem);

#pragma omp parallel for
    for (uint32_t i = 0; i < result.n_elem; i++) {
        result_vec[i] = result(i);
    }

    return result_vec;
}

double plasmon_potcoef_lwl(
    const std::vector<double>& kk, double ls, const system_data& sys) {
    constexpr uint32_t n_int{1};
    double result[n_int] = {0}, error[n_int] = {0};

    /*
     * kk -> (k1, k2)
     */

    plasmon_potcoef_lwl_s s{kk[0], kk[1], ls, sys};

    gsl_function integrands[n_int];

    integrands[0].function = &plasmon_potcoef_lwl_f;
    integrands[0].params   = &s;

    constexpr uint32_t local_ws_size{(1 << 9)};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, M_PI, 0, global_eps, 0, local_ws_size, GSL_INTEG_GAUSS31,
        ws, result, error);

    gsl_integration_workspace_free(ws);
    return result[0] * M_1_PI;
}

std::vector<double> plasmon_potcoef_lwl_v(
    const std::vector<std::vector<double>>& kk_vec,
    double ls,
    const system_data& sys) {
    uint64_t N_total{kk_vec.size()};
    std::vector<double> result(N_total);

#pragma omp parallel for
    for (uint32_t i = 0; i < N_total; i++) {
        result[i] = plasmon_potcoef_lwl(kk_vec[i], ls, sys);
    }

    return result;
}

template <typename T>
void plasmon_invert(typename T::elem_type& v) {
    v = 1.0 / v;
}

template <uint8_t return_part = 0, typename T, typename T2>
auto plasmon_det_f(T2 z, void* params) {
    plasmon_mat_s<T>* s{static_cast<plasmon_mat_s<T>*>(params)};

    s->mat_z_G0.diag().fill(-z);
    s->mat_z_G0.diag() += s->mat_G0.diag();
    s->mat_z_G0.for_each(&plasmon_invert<arma::Mat<T>>);

    s->mat_potcoef = s->mat_kron + s->mat_elem * s->mat_z_G0;

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

template <typename T = std::complex<double>>
void plasmon_fill_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta,
    const arma::Mat<T>& potcoef,
    arma::Mat<T>& mat_elem,
    arma::SpMat<T>& mat_G0) {
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
        for (uint32_t j = 0; j < N_w; j++) {
            for (uint32_t l = 0; l < N_w; l++) {
                for (uint32_t i = 0; i < N_k; i++) {
                    for (uint32_t k = 0; k < N_k; k++) {
                        const std::vector<double> k_v{
                            (1.0 - u0(i)) / u0(i),
                            (1.0 - u0(k)) / u0(k),
                        };

                        mat_elem(j + N_w * i, l + N_w * k) =
                            du0 * du1 * k_v[1] * (1 + std::pow(u1(l), 2)) /
                            std::pow(u0(k) * (1 - std::pow(u1(l), 2)), 2) *
                            potcoef(j + N_w * i, l + N_w * k);

                        if (i == k && j == l) {
                            mat_G0(j + N_w * i, l + N_w * k) =
                                -std::pow(sys.c_hbarc * k_v[1], 2) * 0.5 /
                                sys.m_p;
                        }
                    }
                }
            }
        }
    } else {
#pragma omp parallel for
        for (uint32_t i = 0; i < N_k; i++) {
            for (uint32_t k = 0; k < N_k; k++) {
                const double t_v[2] = {u0(i), u0(k)};

                const std::vector<double> k_v{
                    (1.0 - t_v[0]) / t_v[0],
                    (1.0 - t_v[1]) / t_v[1],
                };

                mat_elem(i, k) =
                    du0 * k_v[1] / std::pow(t_v[1], 2) * potcoef(i, k);

                if (i == k) {
                    mat_G0(i, k) =
                        -std::pow(sys.c_hbarc * k_v[1], 2) * 0.5 / sys.m_p;
                }
            }
        }
    }
}

template <
    std::complex<double> (*green_func)(
        double, double, double, double, const system_data& sys, double),
    typename T          = std::complex<double>,
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
    arma::SpMat<T> mat_kron{
        arma::speye<arma::SpMat<T>>(N_k * N_w, N_k * N_w),
    };

    arma::SpMat<T> mat_G0(N_k * N_w, N_k * N_w);

    // Change every step
    arma::SpMat<T> mat_z_G0(N_k * N_w, N_k * N_w);
    arma::Mat<T> mat_potcoef{
        plasmon_potcoef_mat<plasmon_potcoef<green_func, T>, T>(
            N_k, N_w, mu_e, mu_h, sys, delta),
    };

    if constexpr (print_progress) {
        printf("Initializing\n");
    }

    plasmon_fill_mat<T>(
        N_k, N_w, mu_e, mu_h, sys, delta, mat_potcoef, mat_elem, mat_G0);

    struct plasmon_mat_s<T> s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
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
    return plasmon_det_t<plasmon_green, std::complex<double>, true, true>(
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
    return plasmon_det_t<plasmon_green_ht, std::complex<double>, true, true>(
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
    return plasmon_det_t<plasmon_green, double, false, true>(
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
    return plasmon_det_t<plasmon_green_ht, double, false, true>(
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
    return plasmon_det_t<plasmon_green, std::complex<double>, false, true>(
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
    return plasmon_det_t<plasmon_green_ht, std::complex<double>, false, true>(
        z_vec, N_k, N_w, mu_e, mu_h, sys, delta);
}

template <
    std::complex<double> (*green_func)(
        double, double, double, double, const system_data& sys, double),
    bool check_sign = true,
    typename T      = double>
double plasmon_det_zero_t(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    // Constant
    arma::Mat<T> mat_elem(N_k * N_w, N_k * N_w);
    arma::SpMat<T> mat_kron{
        arma::speye<arma::SpMat<T>>(N_k * N_w, N_k * N_w),
    };

    arma::SpMat<T> mat_G0(N_k * N_w, N_k * N_w);

    // Change every step
    arma::SpMat<T> mat_z_G0(N_k * N_w, N_k * N_w);
    arma::Mat<T> mat_potcoef{
        plasmon_potcoef_mat<plasmon_potcoef<green_func, T>, T>(
            N_k, N_w, mu_e, mu_h, sys, delta),
    };

    plasmon_fill_mat<T>(
        N_k, N_w, mu_e, mu_h, sys, delta, mat_potcoef, mat_elem, mat_G0);

    struct plasmon_mat_s<T> s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    gsl_function funct;
    funct.function = &plasmon_det_f<0, T>;
    funct.params   = &s;

    double z{0};
    double z_min{1e-10}, z_max{-1.5 * sys.get_E_n(0.5)};

    if constexpr (check_sign) {
        if (funct.function(z_max, funct.params) *
                funct.function(z_min, funct.params) >
            0) {
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

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return -z;
}

double plasmon_det_zero(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_zero_t<plasmon_green, true>(
        N_k, N_w, mu_e, mu_h, sys, delta);
}

double plasmon_det_zero_nsc(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_zero_t<plasmon_green, false>(
        N_k, N_w, mu_e, mu_h, sys, delta);
}

double plasmon_det_zero_ht(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_zero_t<plasmon_green_ht, true>(
        N_k, N_w, mu_e, mu_h, sys, delta);
}

double plasmon_det_zero_ht_nsc(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta) {
    return plasmon_det_zero_t<plasmon_green_ht, false>(
        N_k, N_w, mu_e, mu_h, sys, delta);
}

std::vector<double> plasmon_mat_lwl(
    const std::vector<std::vector<double>>& vu_vec,
    const std::vector<std::vector<uint32_t>>& id_vec,
    uint32_t N_u,
    double du,
    double ls,
    double z,
    const system_data& sys) {
    uint64_t N_vu{vu_vec.size()};

    // Constant
    arma::mat mat_elem(N_u, N_u);
    arma::sp_mat mat_kron{arma::speye<arma::sp_mat>(N_u, N_u)};

    arma::sp_mat mat_G0(N_u, N_u);

    // Change every step
    arma::sp_mat mat_z_G0(N_u, N_u);
    arma::mat mat_potcoef(N_u, N_u);

    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    std::vector<double> q_vec(N_u, 1.0);
    q_vec[0]       = 17.0 / 48.0;
    q_vec[1]       = 59.0 / 48.0;
    q_vec[2]       = 43.0 / 48.0;
    q_vec[3]       = 49.0 / 48.0;
    q_vec[N_u - 4] = 49.0 / 48.0;
    q_vec[N_u - 3] = 43.0 / 48.0;
    q_vec[N_u - 2] = 59.0 / 48.0;
    q_vec[N_u - 1] = 17.0 / 48.0;

    mat_kron.diag() = arma::vec(q_vec.data(), N_u, false, true);

// omp_set_num_threads(32);
#pragma omp parallel for
    for (uint64_t i = 0; i < N_vu; i++) {
        const std::vector<double>& t_v{vu_vec[i]};
        const std::vector<uint32_t>& t_id{id_vec[i]};

        const std::vector<double> k_v{(1.0 - t_v[0]) / t_v[0],
                                      (1.0 - t_v[1]) / t_v[1]};

        mat_elem(t_id[0], t_id[1]) = du * q_vec[t_id[1]] * k_v[1] /
                                     std::pow(t_v[1], 2) *
                                     plasmon_potcoef_lwl(k_v, ls, sys);

        mat_G0(t_id[0], t_id[1]) =
            -std::pow(sys.c_hbarc * k_v[1], 2) * 0.5 / sys.m_p;
    }

    struct plasmon_mat_s<double> s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    mat_potcoef = plasmon_det_f<3, double>(z, &s);

    return std::vector<double>(mat_potcoef.begin(), mat_potcoef.end());
}

double plasmon_det_zero_lwl(
    const std::vector<std::vector<double>>& vu_vec,
    const std::vector<std::vector<uint32_t>>& id_vec,
    uint32_t N_u,
    double du,
    double ls,
    const system_data& sys) {
    uint64_t N_vu{vu_vec.size()};

    // Constant
    arma::mat mat_elem(N_u, N_u);
    arma::sp_mat mat_kron{arma::speye<arma::sp_mat>(N_u, N_u)};

    arma::sp_mat mat_G0(N_u, N_u);

    // Change every step
    arma::sp_mat mat_z_G0(N_u, N_u);
    arma::mat mat_potcoef(N_u, N_u);

    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    std::vector<double> q_vec(N_u, 1.0);
    q_vec[0]       = 17.0 / 48.0;
    q_vec[1]       = 59.0 / 48.0;
    q_vec[2]       = 43.0 / 48.0;
    q_vec[3]       = 49.0 / 48.0;
    q_vec[N_u - 4] = 49.0 / 48.0;
    q_vec[N_u - 3] = 43.0 / 48.0;
    q_vec[N_u - 2] = 59.0 / 48.0;
    q_vec[N_u - 1] = 17.0 / 48.0;

    mat_kron.diag() = arma::vec(q_vec);

// omp_set_num_threads(32);
#pragma omp parallel for
    for (uint64_t i = 0; i < N_vu; i++) {
        const std::vector<double>& t_v{vu_vec[i]};
        const std::vector<uint32_t>& t_id{id_vec[i]};

        const std::vector<double> k_v{(1.0 - t_v[0]) / t_v[0],
                                      (1.0 - t_v[1]) / t_v[1]};

        mat_elem(t_id[0], t_id[1]) = du * q_vec[t_id[1]] * k_v[1] /
                                     std::pow(t_v[1], 2) *
                                     plasmon_potcoef_lwl(k_v, ls, sys);

        if (t_id[0] == t_id[1]) {
            mat_G0(t_id[0], t_id[1]) =
                -std::pow(sys.c_hbarc * k_v[1], 2) * 0.5 / sys.m_p;
        }
    }

    struct plasmon_mat_s<double> s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    gsl_function funct;
    funct.function = &plasmon_det_f<0, double>;
    funct.params   = &s;

    double z{0};
    double z_min{-0.5 * sys.get_E_n(0.5)}, z_max{-1.5 * sys.get_E_n(0.5)};
    double f1;

    while (z_max > 1e-14) {
        f1 = funct.function(z_min, funct.params);

        if (f1 < 0) {
            break;
        } else if (std::isnan(f1)) {
            return std::numeric_limits<double>::quiet_NaN();
        } else {
            z_max = z_min;
            z_min = 0.5 * z_max;
        }
    }

    if (funct.function(z_max, funct.params) * f1 > 0) {
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
    return -z;
}

double plasmon_real_potcoef_k_f1(double k, void* params) {
    plasmon_real_potcoef_k_s* s{
        static_cast<plasmon_real_potcoef_k_s*>(params)};

    return -s->sys.c_aEM / s->sys.eps_r * 2 * M_PI * s->sys.c_hbarc *
           (1.0 +
            1.0 / (1.0 + s->sys.c_aEM * (s->sys.m_e + s->sys.m_h) /
                             (2 * M_PI * s->sys.c_hbarc * k * s->sys.eps_r))) *
           gsl_sf_bessel_J0(k * s->x);
}

double plasmon_real_potcoef_k_f2(double k, void* params) {
    plasmon_real_potcoef_k_s* s{
        static_cast<plasmon_real_potcoef_k_s*>(params)};

    std::complex<double> elem{
        plasmon_green(0, k, s->mu_e, s->mu_h, s->sys, 1e-12)};

    return (k * elem.real() - s->sys.c_aEM / s->sys.eps_r * s->sys.c_hbarc) *
           gsl_sf_bessel_J0(k * s->x);
}

double plasmon_real_potcoef_k_f3(double k, void* params) {
    plasmon_real_potcoef_k_s* s{
        static_cast<plasmon_real_potcoef_k_s*>(params)};

    std::complex<double> elem{plasmon_green(0, k, s->mu_e, s->mu_h, s->sys)};

    return (k * elem.real() - s->sys.c_aEM / s->sys.eps_r * s->sys.c_hbarc) /
           std::sqrt(M_PI * k * s->x);
}

double plasmon_real_pot(
    double x, double mu_e, double mu_h, const system_data& sys) {
    constexpr uint32_t n_int{2};
    constexpr uint32_t local_ws_size{1 << 7};
    constexpr uint32_t t_size{1 << 9};

    double result[n_int] = {1}, error[n_int] = {0};
    constexpr uint32_t n_sum{1 << 7};

    double result_sum = {0.0};
    double last_zero  = {0.0};
    double t[n_sum];

    plasmon_real_potcoef_k_s s{x, mu_e, mu_h, sys};

    gsl_function integrands[n_int];

    integrands[0].function = &plasmon_real_potcoef_k_f2;
    integrands[0].params   = &s;

    integrands[1].function = &plasmon_real_potcoef_k_f3;
    integrands[1].params   = &s;

    gsl_integration_qawo_table* qawo_table[2] = {
        gsl_integration_qawo_table_alloc(x, 1, GSL_INTEG_COSINE, t_size),
        gsl_integration_qawo_table_alloc(x, 1, GSL_INTEG_SINE, t_size),
    };

    gsl_integration_workspace* ws[2] = {
        gsl_integration_workspace_alloc(local_ws_size),
        gsl_integration_workspace_alloc(local_ws_size)};

    for (uint32_t i = 0; i < n_sum; i++) {
        double temp = gsl_sf_bessel_zero_J0(i + 1) / x;

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
        integrands + 1, last_zero, 0, t_size, ws[0], ws[1], qawo_table[0],
        result + 1, error + 1);

    result_sum += result[1];

    gsl_integration_workspace_free(ws[0]);
    gsl_integration_workspace_free(ws[1]);

    gsl_integration_qawo_table_free(qawo_table[0]);
    gsl_integration_qawo_table_free(qawo_table[1]);

    return sys.c_aEM * sys.c_hbarc / (x * sys.eps_r) + result_sum;
}

std::vector<double> plasmon_real_pot_v(
    const std::vector<double>& x_vec,
    double mu_e,
    double mu_h,
    const system_data& sys) {
    uint64_t N_x{x_vec.size()};
    std::vector<double> output(N_x);

#pragma omp parallel for
    for (uint64_t i = 0; i < N_x; i++) {
        output[i] = plasmon_real_pot(x_vec[i], mu_e, mu_h, sys);
    }

    return output;
}
