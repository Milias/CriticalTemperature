#include "biexcitons.h"

double biexciton_Delta_th_f(double th, biexciton_pot_s* s) {
    /*
     * The result is the same for + 2 x cos(th) or - 2 x cos(th).
     */
    return std::exp(
        -s->param_alpha *
        std::sqrt(
            1 + s->param_x2 * s->param_x2 - 2 * s->param_x2 * std::cos(th)));
}

result_s<1> biexciton_Delta_th(double param_alpha, double param_x) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{param_alpha, 0.0, 0.0, param_x};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<biexciton_pot_s, biexciton_Delta_th_f>;
    integrands[0].params = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, M_PI, 0, local_eps, local_ws_size, GSL_INTEG_GAUSS31,
        ws, result.value, result.error);

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_Delta_r_f(double param_x, biexciton_pot_s* s) {
    return param_x * std::exp(-s->param_alpha * param_x) *
           biexciton_Delta_th(s->param_alpha, param_x).total_value();
}

result_s<1> biexciton_Delta_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<biexciton_pot_s, biexciton_Delta_r_f>;
    integrands[0].params = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qagiu(
        integrands, 0.0, 0.0, local_eps, local_ws_size, ws, result.value,
        result.error);

    /*
     * Factor of 2 because of integration [0, pi).
     */
    result.value[0] *= 4 * M_1_PI * std::pow(r_BA / sys.a0, 2);

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_J_r_f(double param_x, biexciton_pot_s* s) {
    if (param_x == 0) {
        return 0.0;
    }

    return param_x / (1 + param_x) * std::exp(-2 * s->param_alpha * param_x) *
           gsl_sf_ellint_Kcomp(
               2 * std::sqrt(param_x) / (1 + param_x), GSL_PREC_DOUBLE);
}

result_s<2> biexciton_J_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{2};
    biexciton_pot_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_J_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    double int_pts[] = {0.0, 1.0, 2.0};
    gsl_integration_qagp(
        integrands, int_pts, sizeof(int_pts) / sizeof(double), 0.0, local_eps,
        local_ws_size, ws, result.value, result.error);

    gsl_integration_qagiu(
        integrands, int_pts[2], 0.0, local_eps, local_ws_size, ws,
        result.value + 1, result.error + 1);

    result.value[0] *= -8 * M_1_PI * std::pow(sys.c_aEM / sys.eps_r, 2) *
                       sys.m_p * r_BA / sys.a0;

    result.value[1] *= -8 * M_1_PI * std::pow(sys.c_aEM / sys.eps_r, 2) *
                       sys.m_p * r_BA / sys.a0;

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_Jp_r2_f(double param_x2, biexciton_pot_s* s) {
    return param_x2 / (s->param_x1 + param_x2) *
           biexciton_Delta_th(2 * s->param_alpha, param_x2).total_value() *
           gsl_sf_ellint_Kcomp(
               2 * std::sqrt(param_x2 * s->param_x1) /
                   (s->param_x1 + param_x2),
               GSL_PREC_DOUBLE);
}

result_s<2> biexciton_Jp_r2(double param_alpha, double param_x1) {
    constexpr uint32_t n_int{2};
    biexciton_pot_s s{param_alpha, 0.0, param_x1, 0.0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_Jp_r2_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 10)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    double int_pts[] = {0, param_x1, 2 * param_x1};
    gsl_integration_qagp(
        integrands, int_pts, sizeof(int_pts) / sizeof(double), 0.0, local_eps,
        local_ws_size, ws, result.value, result.error);

    gsl_integration_qagiu(
        integrands, int_pts[2], 0.0, local_eps, local_ws_size, ws,
        result.value + 1, result.error + 1);

    gsl_integration_workspace_free(ws);

    return result;
}

double biexciton_Jp_r_f(double param_x1, biexciton_pot_s* s) {
    return param_x1 * std::exp(-2 * s->param_alpha * param_x1) *
           biexciton_Jp_r2(s->param_alpha, param_x1).total_value();
}

result_s<1> biexciton_Jp_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_Jp_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 6)};
    constexpr double local_eps{1e-6};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qagiu(
        integrands, 0.0, 0.0, local_eps, local_ws_size, ws, result.value,
        result.error);

    result.value[0] *= 32 * std::pow(r_BA / sys.a0, 3) * M_1_PI * M_1_PI *
                       sys.c_aEM * sys.c_hbarc / (sys.a0 * sys.eps_r);

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_K_r_f(double param_x, biexciton_pot_s* s) {
    return std::exp(-s->param_alpha * param_x) *
           biexciton_Delta_th(s->param_alpha, param_x).total_value();
}

result_s<1> biexciton_K_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_K_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qagiu(
        integrands, 0.0, 0.0, local_eps, local_ws_size, ws, result.value,
        result.error);

    result.value[0] *= -4 * M_1_PI * sys.c_aEM * sys.c_hbarc * r_BA /
                       (sys.eps_r * std::pow(sys.a0, 2));

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_Kp_th2_f(double param_th, biexciton_pot_s* s) {
    return std::exp(
               -s->param_alpha * std::sqrt(
                                     1 + s->param_x2 * s->param_x2 -
                                     2 * s->param_x2 * std::cos(param_th))) /
           std::sqrt(
               s->param_x1 * s->param_x1 + s->param_x2 * s->param_x2 -
               2 * s->param_x1 * s->param_x2 *
                   std::cos(s->param_th1 - param_th));
}

result_s<1> biexciton_Kp_th2(
    double param_alpha, double param_th1, double param_x1, double param_x2) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{param_alpha, param_th1, param_x1, param_x2};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_Kp_th2_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 6)};
    constexpr double local_eps{1e-6};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, 2 * M_PI, 0, local_eps, local_ws_size,
        GSL_INTEG_GAUSS31, ws, result.value, result.error);

    gsl_integration_workspace_free(ws);

    return result;
}

double biexciton_Kp_th1_f(double param_th, biexciton_pot_s* s) {
    return std::exp(
               -s->param_alpha * std::sqrt(
                                     1 + s->param_x1 * s->param_x1 -
                                     2 * s->param_x1 * std::cos(param_th))) *
           biexciton_Kp_th2(s->param_alpha, param_th, s->param_x1, s->param_x2)
               .total_value();
}

result_s<1> biexciton_Kp_th1(
    double param_alpha, double param_x1, double param_x2) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{param_alpha, 0.0, param_x1, param_x2};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_Kp_th1_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 6)};
    constexpr double local_eps{1e-6};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, 2 * M_PI, 0, local_eps, local_ws_size,
        GSL_INTEG_GAUSS31, ws, result.value, result.error);

    gsl_integration_workspace_free(ws);

    return result;
}

double biexciton_Kp_r1_f(double r1, biexciton_pot_s* s) {
    return r1 * std::exp(-s->param_alpha * r1) *
           biexciton_Kp_th1(s->param_alpha, r1, s->param_x2).total_value();
}

result_s<2> biexciton_Kp_r1(double param_alpha, double param_x2) {
    constexpr uint32_t n_int{2};
    biexciton_pot_s s{param_alpha, 0.0, 0.0, param_x2};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_Kp_r1_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 6)};
    constexpr double local_eps{1e-6};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    double int_pts[] = {0.0, s.param_x2, 2.0 * s.param_x2};
    gsl_integration_qagp(
        integrands, int_pts, sizeof(int_pts) / sizeof(double), 0.0, local_eps,
        local_ws_size, ws, result.value, result.error);

    gsl_integration_qagiu(
        integrands, int_pts[2], 0.0, local_eps, local_ws_size, ws,
        result.value + 1, result.error + 1);

    gsl_integration_workspace_free(ws);

    /*
    printf(
        "Kp_r1: [%f] %f (%.2e)\n", param_x2, result.total_value(),
        result.total_error());
    */

    return result;
}

double biexciton_Kp_r_f(double r2, biexciton_pot_s* s) {
    return r2 * std::exp(-s->param_alpha * r2) *
           biexciton_Kp_r1(s->param_alpha, r2).total_value();
}

result_s<1> biexciton_Kp_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_pot_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_pot_s, biexciton_Kp_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 6)};
    constexpr double local_eps{1e-6};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qagiu(
        integrands, 0.0, 0.0, local_eps, local_ws_size, ws, result.value,
        result.error);

    gsl_integration_workspace_free(ws);

    result.value[0] *= 4 * std::pow(r_BA / sys.a0, 3) * M_1_PI * M_1_PI *
                       sys.c_aEM * sys.c_hbarc / (sys.a0 * sys.eps_r);

    return result;
}

template <size_t return_type = 0>
auto biexciton_eff_pot(double r_BA, const system_data& sys) {
    /*
     * return_type:
     *      0 => returns a result_s<7>
     *      1 => returns a double, potential +.
     *      2 => returns a double, potential -.
     */
    result_s<1> r_D{biexciton_Delta_r(r_BA, sys)};
    result_s<2> r_J{biexciton_J_r(r_BA, sys)};
    result_s<1> r_Jp{biexciton_Jp_r(r_BA, sys)};
    result_s<1> r_K{biexciton_K_r(r_BA, sys)};
    result_s<1> r_Kp{biexciton_Kp_r(r_BA, sys)};

    double D{r_D.total_value()}, J{r_J.total_value()}, Jp{r_Jp.total_value()},
        K{r_K.total_value()}, Kp{r_Kp.total_value()};

    if constexpr (return_type == 0) {
        result_s<7> pot;

        pot.value[0] = D;
        pot.error[0] = r_D.total_error();
        pot.value[1] = J;
        pot.error[1] = r_J.total_error();
        pot.value[2] = Jp;
        pot.error[2] = r_Jp.total_error();
        pot.value[3] = K;
        pot.error[3] = r_K.total_error();
        pot.value[4] = Kp;
        pot.error[4] = r_Kp.total_error();

        pot.value[5] = sys.c_aEM * sys.c_hbarc / r_BA / sys.eps_r +
                       (2 * J + Jp + 2 * D * K + Kp) / (1 + D * D);
        pot.value[6] = sys.c_aEM * sys.c_hbarc / r_BA / sys.eps_r +
                       (2 * J + Jp - 2 * D * K - Kp) / (1 - D * D);

        return pot;

    } else if (return_type == 1) {
        return sys.c_aEM * sys.c_hbarc / r_BA / sys.eps_r +
               (2 * J + Jp + 2 * D * K + Kp) / (1 + D * D);
    } else if (return_type == 2) {
        return sys.c_aEM * sys.c_hbarc / r_BA / sys.eps_r +
               (2 * J + Jp - 2 * D * K - Kp) / (1 - D * D);
    }
}

template <
    size_t N,
    result_s<N> (*F)(double, const system_data&),
    bool print_progress = false>
std::vector<result_s<N>> biexciton_omp_f(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    std::vector<result_s<N>> result_vec(r_BA_vec.size());

    uint32_t num{0};

#pragma omp parallel for
    for (uint32_t i = 0; i < r_BA_vec.size(); i++) {
        result_vec[i] = F(r_BA_vec[i], sys);

        if constexpr (print_progress) {
            num++;
            printf(
                "[%d/%ld] x: %f, val: %f, err: %e\n", num, r_BA_vec.size(),
                r_BA_vec[i], result_vec[i].total_value(),
                result_vec[i].total_error());
        }
    }

    return result_vec;
}

std::vector<result_s<1>> biexciton_Delta_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    return biexciton_omp_f<1, biexciton_Delta_r, true>(r_BA_vec, sys);
}

std::vector<result_s<2>> biexciton_J_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    return biexciton_omp_f<2, biexciton_J_r, true>(r_BA_vec, sys);
}

std::vector<result_s<1>> biexciton_Jp_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    return biexciton_omp_f<1, biexciton_Jp_r, true>(r_BA_vec, sys);
}

std::vector<result_s<1>> biexciton_K_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    return biexciton_omp_f<1, biexciton_K_r, true>(r_BA_vec, sys);
}

std::vector<result_s<1>> biexciton_Kp_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    return biexciton_omp_f<1, biexciton_Kp_r, true>(r_BA_vec, sys);
}

std::vector<result_s<7>> biexciton_eff_pot_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys) {
    return biexciton_omp_f<7, biexciton_eff_pot<0>, true>(r_BA_vec, sys);
}

struct biexciton_eff_pot_s {
    const system_data& sys;
    gsl_interp_accel* acc = nullptr;
    gsl_spline* spline    = nullptr;

    double xmin{0}, xmax{0};

    void interp_pot(
        const std::vector<double>& x_vec, const std::vector<double>& pot_vec) {
        acc    = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_steffen, x_vec.size());
        xmin   = x_vec.front();
        xmax   = x_vec.back();

        gsl_spline_init(
            spline, &x_vec.front(), &pot_vec.front(), x_vec.size());
    }

    ~biexciton_eff_pot_s() {
        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);
    }

    double operator()(double x) {
        double r{0};

        if (x >= xmin && x <= xmax) {
            r = gsl_spline_eval(spline, x, acc);
        } else {
            r = biexciton_eff_pot<1>(x, sys);
        }

        return r;
    };
};

std::vector<result_s<1>> biexciton_eff_pot_interp_vec(
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    const std::vector<double>& x_interp_vec,
    const system_data& sys) {
    biexciton_eff_pot_s pot{sys};

    pot.interp_pot(x_vec, pot_vec);

    std::vector<result_s<1>> r(x_interp_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_interp_vec.size(); i++) {
        r[i].value[0] = pot(x_interp_vec[i]);
    }

    return r;
}

std::vector<double> biexciton_wf(
    double E,
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    uint32_t n_steps,
    const system_data& sys) {
    biexciton_eff_pot_s pot{sys};
    pot.interp_pot(x_vec, pot_vec);

    auto [f_vec, t_vec] = wf_gen_s_r_t<biexciton_eff_pot_s>(
        E, x_vec.front(), sys.c_alpha_bexc, x_vec.back(), n_steps, pot);
    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double biexciton_be(
    double E_min,
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    const system_data& sys) {
    biexciton_eff_pot_s pot{sys};
    pot.interp_pot(x_vec, pot_vec);

    return wf_gen_E_t<biexciton_eff_pot_s, 0>(
        E_min, x_vec.front(), sys.c_alpha_bexc, pot, x_vec.back());
}

