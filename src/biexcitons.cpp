#include "biexcitons.h"

double biexciton_Delta_th_f(double th, biexciton_Delta_th_s* s) {
    /*
     * The result is the same for + 2 x cos(th) or - 2 x cos(th).
     */
    // COZ_PROGRESS_NAMED("biexciton_Delta_th_f");
    return std::exp(
        -s->param_alpha *
        std::sqrt(
            1 + s->param_x * s->param_x - 2 * s->param_x * std::cos(th)));
}

result_s<1> biexciton_Delta_th(double param_alpha, double param_x) {
    constexpr uint32_t n_int{1};
    biexciton_Delta_th_s s{param_alpha, param_x};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<biexciton_Delta_th_s, biexciton_Delta_th_f>;
    integrands[0].params = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, M_PI, 0, local_eps, local_ws_size, GSL_INTEG_GAUSS31,
        ws, result.value, result.error);

    gsl_integration_workspace_free(ws);

    /*
     * This is half of the final value, multiplied
     * times 2 in biexciton_Delta_r.
     */
    return result;
}

double biexciton_Delta_r_f(double param_x, biexciton_Delta_r_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Delta_r_f");
    return param_x * std::exp(-s->param_alpha * param_x) *
           biexciton_Delta_th(s->param_alpha, param_x).total_value();
}

result_s<1> biexciton_Delta_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_Delta_r_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<biexciton_Delta_r_s, biexciton_Delta_r_f>;
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
    result.value[0] *= 2 * M_1_PI * std::pow(r_BA / sys.a0, 2);

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_J_r_f(double param_x, biexciton_J_r_s* s) {
    if (param_x == 0) {
        return 0.0;
    }

    const double abs_val{param_x + 1};

    // COZ_PROGRESS_NAMED("biexciton_J_r_f");
    return param_x / abs_val * std::exp(-2 * s->param_alpha * param_x) *
           gsl_sf_ellint_Kcomp(
               2 * std::sqrt(param_x) / abs_val, GSL_PREC_DOUBLE);
}

result_s<2> biexciton_J_r(double r_BA, const system_data& sys) {
    if (r_BA < 1e-5) {
        r_BA = 1e-5;
    }

    constexpr uint32_t n_int{2};
    biexciton_J_r_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_J_r_s, biexciton_J_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    double int_pts[] = {0.0, 1.0, 2.0};
    gsl_integration_qagp(
        integrands, int_pts, sizeof(local_ws_size) / sizeof(double), 0.0,
        local_eps, local_ws_size, ws, result.value, result.error);

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

double biexciton_Jp_th_f(double param_th, biexciton_Jp_th_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Jp_th_f");
    return std::exp(
        -2 * s->param_alpha *
        std::sqrt(
            1 + s->param_x1 * s->param_x1 +
            2 * s->param_x1 * std::cos(param_th)));
}

result_s<1> biexciton_Jp_th(double param_alpha, double param_x1) {
    param_x1 = param_x1 < 1e-5 ? 1e-5 : param_x1;

    constexpr uint32_t n_int{1};
    biexciton_Jp_th_s s{param_alpha, param_x1};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<biexciton_Jp_th_s, biexciton_Jp_th_f>;
    integrands[0].params = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, M_PI, 0, local_eps, local_ws_size, GSL_INTEG_GAUSS31,
        ws, result.value, result.error);

    gsl_integration_workspace_free(ws);

    /*
     * This is half of the final value, multiplied
     * times 2 in biexciton_Jp_r1.
     */

    return result;
}

double biexciton_Jp_r2_f(double param_x2, biexciton_Jp_r2_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Jp_r2_f");
    return param_x2 / (1 + param_x2) *
           std::exp(-2 * s->param_alpha * s->param_x1 * param_x2) *
           gsl_sf_ellint_Kcomp(
               2 * std::sqrt(param_x2) / (1 + param_x2), GSL_PREC_DOUBLE);
}

result_s<2> biexciton_Jp_r2(double param_alpha, double param_x1) {
    constexpr uint32_t n_int{2};
    biexciton_Jp_r2_s s{param_alpha, param_x1};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<biexciton_Jp_r2_s, biexciton_Jp_r2_f>;
    integrands[0].params = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    double int_pts[] = {0.0, 1.0, 2.0};
    gsl_integration_qagp(
        integrands, int_pts, sizeof(local_ws_size) / sizeof(double), 0.0,
        local_eps, local_ws_size, ws, result.value, result.error);

    gsl_integration_qagiu(
        integrands, int_pts[2], 0.0, local_eps, local_ws_size, ws,
        result.value + 1, result.error + 1);

    /*
     * This is a quarter of the final value, multiplied
     * times 4 in biexciton_Jp_r1.
     */

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_Jp_r_f(double param_x1, biexciton_Jp_r_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Jp_r_f");

    return param_x1 * param_x1 *
           biexciton_Jp_th(s->param_alpha, param_x1).total_value() *
           biexciton_Jp_r2(s->param_alpha, param_x1).total_value();
}

result_s<1> biexciton_Jp_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_J_r_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_Jp_r_s, biexciton_Jp_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

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

double biexciton_K_r_f(double param_x, biexciton_K_r_s* s) {
    /*
     * Reusing the angular integral for computing Delta.
     */
    // COZ_PROGRESS_NAMED("biexciton_K_r_f");
    return std::exp(-s->param_alpha * param_x) *
           biexciton_Delta_th(s->param_alpha, param_x).total_value();
}

result_s<1> biexciton_K_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_K_r_s s{r_BA / sys.a0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_K_r_s, biexciton_K_r_f>;
    integrands[0].params   = &s;

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
    result.value[0] *= -4 * M_1_PI * sys.c_aEM * sys.c_hbarc * r_BA /
                       (sys.eps_r * std::pow(sys.a0, 2));

    gsl_integration_workspace_free(ws);
    return result;
}

double biexciton_Kp_th2_f(double th2, biexciton_Kp_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Kp_th2_f");
    return std::exp(
               -s->param_alpha * std::sqrt(
                                     1 + s->param_x2 * s->param_x2 -
                                     2 * s->param_x2 * std::cos(th2))) /
           std::sqrt(
               s->param_x1 * s->param_x1 + s->param_x2 * s->param_x2 -
               2 * s->param_x1 * s->param_x2 * std::cos(s->param_th1 - th2));
}

result_s<1> biexciton_Kp_th2(
    double param_alpha, double param_th1, double param_x1, double param_x2) {
    constexpr uint32_t n_int{1};
    biexciton_Kp_s s{param_alpha, param_th1, param_x1, param_x2};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_Kp_s, biexciton_Kp_th2_f>;
    integrands[0].params   = &s;

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

double biexciton_Kp_th1_f(double th1, biexciton_Kp_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Kp_th1_f");
    return std::exp(
               -s->param_alpha * (std::sqrt(
                                     1 + s->param_x1 * s->param_x1 -
                                     2 * s->param_x1 * std::cos(th1)))) *
           biexciton_Kp_th2(s->param_alpha, th1, s->param_x1, s->param_x2)
               .total_value();
}

result_s<1> biexciton_Kp_th1(
    double param_alpha, double param_x1, double param_x2) {
    constexpr uint32_t n_int{1};
    biexciton_Kp_s s{param_alpha, 0.0, param_x1, param_x2};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_Kp_s, biexciton_Kp_th1_f>;
    integrands[0].params   = &s;

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

double biexciton_Kp_r1_f(double r1, biexciton_Kp_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Kp_r1_f");
    return r1 *
           biexciton_Kp_th1(s->param_alpha, r1, s->param_x2).total_value();
}

result_s<2> biexciton_Kp_r1(double param_alpha, double param_x2) {
    constexpr uint32_t n_int{2};
    biexciton_Kp_s s{param_alpha, 0.0, 0.0, param_x2};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_Kp_s, biexciton_Kp_r1_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-10};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    double int_pts[] = {0.0, s.param_x2, 2.0 * s.param_x2};
    gsl_integration_qagp(
        integrands, int_pts, sizeof(local_ws_size) / sizeof(double), 0.0,
        local_eps, local_ws_size, ws, result.value, result.error);

    gsl_integration_qagiu(
        integrands, int_pts[2], 0.0, local_eps, local_ws_size, ws,
        result.value + 1, result.error + 1);

    gsl_integration_workspace_free(ws);

    return result;
}

double biexciton_Kp_r_f(double r2, biexciton_Kp_s* s) {
    // COZ_PROGRESS_NAMED("biexciton_Kp_r_f");
    return r2 * biexciton_Kp_r1(s->param_alpha, r2).total_value();
}

result_s<1> biexciton_Kp_r(double r_BA, const system_data& sys) {
    constexpr uint32_t n_int{1};
    biexciton_Kp_s s{r_BA / sys.a0, 0.0, 0.0, 0.0};
    result_s<n_int> result;

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<biexciton_Kp_s, biexciton_Kp_r_f>;
    integrands[0].params   = &s;

    constexpr size_t local_ws_size{(1 << 6)};
    constexpr double local_eps{1e-6};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qagiu(
        integrands, 0.0, 0.0, local_eps, local_ws_size, ws, result.value,
        result.error);

    gsl_integration_workspace_free(ws);

    result.value[0] *= 16 * std::pow(r_BA / sys.a0, 3) * M_1_PI * M_1_PI *
                       sys.c_aEM * sys.c_hbarc / (sys.a0 * sys.eps_r);

    return result;
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
