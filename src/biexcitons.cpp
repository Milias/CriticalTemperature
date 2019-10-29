#include "biexcitons.h"

double biexciton_eqst_u_f(double mu_e, biexciton_eqst_s* s) {
    return mu_e + s->sys.get_mu_h(mu_e) - s->be_exc - 0.5 * s->be_biexc +
           0.5 * logExp(s->u) / s->sys.beta;
}

result_s<1> biexciton_eqst_u(
    double u,
    double be_exc,
    double be_biexc,
    double mu_e_lim,
    const system_data& sys) {
    constexpr uint32_t local_max_iter{1 << 5};
    constexpr double local_eps{1e-8};

    biexciton_eqst_s params{0.0,
                            be_exc,
                            be_biexc,
                            sys,
                            std::isnan(mu_e_lim)
                                ? sys.exc_mu_val(be_exc + 0.5 * be_biexc)
                                : mu_e_lim,
                            u};

    double z, z_min{params.mu_e_lim},
        z_max{params.mu_e_lim < 0 ? z_min : -z_min};

    gsl_function funct;
    funct.function = &templated_f<biexciton_eqst_s, biexciton_eqst_u_f>;
    funct.params   = &params;

    double f_min{funct.function(z_min, funct.params)}, f_max;

    if (std::abs(f_min) < local_eps) {
        return params.mu_e_lim;
    }

    for (uint32_t i = 0; i <= local_max_iter; i++) {
        z_min = z_max;
        z_max *= 2;

        f_max = funct.function(z_max, funct.params);

        if (f_min * f_max < 0) {
            break;
        } else if (std::isnan(f_max)) {
            return result_s<1>::nan();
        } else {
            f_min = f_max;
        }
    }

    if (f_min * f_max > 0) {
        return params.mu_e_lim;
    }

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* s            = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &funct, z_max, z_min);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < local_max_iter; iter++) {
        status = gsl_root_fsolver_iterate(s);
        z      = gsl_root_fsolver_root(s);
        z_min  = gsl_root_fsolver_x_lower(s);
        z_max  = gsl_root_fsolver_x_upper(s);

        status =
            gsl_root_test_residual(funct.function(z, funct.params), local_eps);
    }

    gsl_root_fsolver_free(s);

    result_s<1> r(z);
    // r.error[0] = funct.function(z, funct.params);

    return r;
}

double biexciton_eqst_c_f(double u, biexciton_eqst_s* s) {
    double mu_e{
        biexciton_eqst_u(u, s->be_exc, s->be_biexc, s->mu_e_lim, s->sys)
            .total_value()};
    double mu_h{s->sys.get_mu_h(mu_e)};
    double mu_exc{mu_e + mu_h};

    double n_q{s->sys.density_ideal(mu_e)};
    double n_exc{s->sys.density_exc(mu_exc, s->be_exc)};
    double n_biexc{s->sys.density_exc2_u(u)};

    return n_q + n_exc + 2 * n_biexc - s->n_gamma;
}

result_s<2> biexciton_eqst_c(
    double n_gamma,
    double be_exc,
    double be_biexc,
    double u_init,
    const system_data& sys) {
    constexpr size_t local_max_iter{1 << 5};
    biexciton_eqst_s s{n_gamma, be_exc, be_biexc, sys,
                       sys.exc_mu_val(be_exc + 0.5 * be_biexc)};

    double z{u_init}, z0;

    gsl_function_fdf funct;
    funct.f      = &templated_f<biexciton_eqst_s, biexciton_eqst_c_f>;
    funct.df     = &templated_df<biexciton_eqst_s, biexciton_eqst_c_f>;
    funct.fdf    = &templated_fdf<biexciton_eqst_s, biexciton_eqst_c_f>;
    funct.params = &s;

    const gsl_root_fdfsolver_type* solver_type = gsl_root_fdfsolver_steffenson;
    gsl_root_fdfsolver* solver = gsl_root_fdfsolver_alloc(solver_type);

    gsl_root_fdfsolver_set(solver, &funct, z);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < local_max_iter; iter++) {
        double f_z{funct.f(z, &s)};
        z0     = z;
        status = gsl_root_fdfsolver_iterate(solver);
        z      = gsl_root_fdfsolver_root(solver);
        status = gsl_root_test_residual(f_z, 1e-6);

        /*
        printf(
            "[%s] iter: %d, z: %f, z0: %f, f_z: %e\n", __func__, iter, z, z0,
            f_z);
        */
    }

    gsl_root_fdfsolver_free(solver);

    result_s<2> r;
    r.value[0] =
        biexciton_eqst_u(z, be_exc, be_biexc, s.mu_e_lim, sys).total_value();
    r.value[1] = z;

    return r;
}

std::vector<double> biexciton_eqst_c_vec(
    const std::vector<double>& n_gamma_T_vec,
    double be_exc,
    double be_biexc,
    double u_init,
    const system_data& sys) {
    std::vector<double> r(n_gamma_T_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < n_gamma_T_vec.size(); i += 2) {
        system_data sys_mod(
            sys.dl_m_e, sys.dl_m_h, sys.eps_r, n_gamma_T_vec[i + 1]);
        result_s<2> mu_e_u{biexciton_eqst_c(
            n_gamma_T_vec[i], be_exc, be_biexc, u_init, sys_mod)};

        r[i]     = mu_e_u.value[0];
        r[i + 1] = mu_e_u.value[1];
    }

    return r;
}

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

    result.value[0] *= -8 * M_1_PI * sys.c_aEM / sys.eps_r * sys.c_hbarc *
                       r_BA / (sys.a0 * sys.a0);
    result.value[1] *= -8 * M_1_PI * sys.c_aEM / sys.eps_r * sys.c_hbarc *
                       r_BA / (sys.a0 * sys.a0);

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

struct biexciton_eff_pot_interp_s {
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

    ~biexciton_eff_pot_interp_s() {
        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);
    }

    double operator()(double x) const {
        double r{0};

        if (x >= xmin && x <= xmax) {
            r = gsl_spline_eval(spline, x, acc);
        } else {
            r = biexciton_eff_pot<1>(x, sys);
        }

        return r;
    };
};

std::vector<double> biexciton_eff_pot_interp_vec(
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    const std::vector<double>& x_interp_vec,
    const system_data& sys) {
    biexciton_eff_pot_interp_s pot{sys};

    pot.interp_pot(x_vec, pot_vec);

    std::vector<double> r(x_interp_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_interp_vec.size(); i++) {
        r[i] = pot(x_interp_vec[i]);
    }

    return r;
}

struct biexciton_pot_r6_s {
    const system_data& sys;
    double param_c6;

    double operator()(double x) const { return -param_c6 / std::pow(x, 6); };
};

std::vector<double> biexciton_pot_r6_vec(
    double be_cou, const std::vector<double>& x_vec, const system_data& sys) {
    constexpr double pol_a2{21.0 / 256.0};

    double param_c6{
        24 / std::pow(be_cou * be_cou * sys.m_p, 3) *
        std::pow(pol_a2 * sys.eps_r * std::pow(sys.c_hbarc, 3), 2)};

    biexciton_pot_r6_s pot{sys, param_c6};

    std::vector<double> r(x_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_vec.size(); i++) {
        r[i] = pot(x_vec[i]);
    }

    return r;
}

std::vector<double> biexciton_wf_hl(
    double E,
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    uint32_t n_steps,
    const system_data& sys) {
    biexciton_eff_pot_interp_s pot{sys};
    pot.interp_pot(x_vec, pot_vec);

    auto [f_vec, t_vec] = wf_gen_s_r_t<biexciton_eff_pot_interp_s>(
        E, x_vec.front(), sys.c_alpha_bexc, x_vec.back(), n_steps, pot);

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

std::vector<double> biexciton_wf_r6(
    double be_biexc,
    double be_cou,
    double r_min,
    double r_max,
    uint32_t n_steps,
    const system_data& sys) {
    constexpr double pol_a2{21.0 / 256.0};

    double param_c6{
        24 / std::pow(be_cou * be_cou * sys.m_p, 3) *
        std::pow(pol_a2 * sys.eps_r * std::pow(sys.c_hbarc, 3), 2)};

    biexciton_pot_r6_s pot{sys, param_c6};

    auto [f_vec, t_vec] = wf_gen_s_r_t<biexciton_pot_r6_s>(
        be_biexc, r_min, sys.c_alpha_bexc, r_max, n_steps, pot);

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double biexciton_be_hl(
    double E_min,
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    const system_data& sys) {
    biexciton_eff_pot_interp_s pot{sys};
    pot.interp_pot(x_vec, pot_vec);

    return wf_gen_E_t<biexciton_eff_pot_interp_s>(
        E_min, x_vec.front(), sys.c_alpha_bexc, pot, x_vec.back());
}

double biexciton_be_r6(
    double E_min, double be_cou, double r_min, const system_data& sys) {
    constexpr double pol_a2{21.0 / 256.0};

    double param_c6{
        24 / std::pow(be_cou * be_cou * sys.m_p, 3) *
        std::pow(pol_a2 * sys.eps_r * std::pow(sys.c_hbarc, 3), 2)};

    biexciton_pot_r6_s pot{sys, param_c6};

    return wf_gen_E_t<biexciton_pot_r6_s>(E_min, r_min, sys.c_alpha_bexc, pot);
}

template <class pot_s>
double biexciton_rmin_f(double r_min, biexciton_rmin_s<pot_s>* s) {
    double r{wf_gen_E_t<pot_s>(s->E_min, r_min, s->sys.c_alpha_bexc, s->pot)};

    if (std::isnan(r)) {
        return s->E_min;
    }

    return r - s->be_biexc;
}

double biexciton_rmin_r6(
    double E_min, double be_cou, double be_biexc, const system_data& sys) {
    using pot_s = biexciton_pot_r6_s;
    constexpr double pol_a2{21.0 / 256.0};

    double param_c6{
        24 / std::pow(be_cou * be_cou * sys.m_p, 3) *
        std::pow(pol_a2 * sys.eps_r * std::pow(sys.c_hbarc, 3), 2)};

    pot_s pot{sys, param_c6};

    constexpr uint32_t local_max_iter{1 << 7};

    biexciton_rmin_s<pot_s> params{E_min, be_biexc, pot, sys};
    double z_min{1.0}, z_max{10.0}, z;

    gsl_function funct;
    funct.function =
        &templated_f<biexciton_rmin_s<pot_s>, biexciton_rmin_f<pot_s>>;
    funct.params = &params;

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* s            = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < local_max_iter; iter++) {
        status = gsl_root_fsolver_iterate(s);
        z      = gsl_root_fsolver_root(s);
        z_min  = gsl_root_fsolver_x_lower(s);
        z_max  = gsl_root_fsolver_x_upper(s);

        // status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
        status =
            gsl_root_test_residual(funct.function(z, &params), global_eps);
        printf(
            "[%d] iterating -- %.16f (%f, %f), %.2e\n", iter, z, z_min, z_max,
            funct.function(z, &params));
    }

    gsl_root_fsolver_free(s);
    return z;
}

std::vector<double> biexciton_be_r6_vec(
    double E_min,
    double be_cou,
    const std::vector<double>& r_min_vec,
    const system_data& sys) {
    std::vector<double> r(r_min_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < r_min_vec.size(); i++) {
        r[i] = biexciton_be_r6(E_min, be_cou, r_min_vec[i], sys);
    }

    return r;
}

struct biexciton_pot_lj_s {
    double param_c12;
    double param_c6;
    const system_data& sys;

    static double get_c6(double be_cou, const system_data& sys) {
        constexpr double pol_a2{21.0 / 256};

        return 24 / std::pow(be_cou * sys.m_p, 2) / sys.m_p *
               std::pow(pol_a2 * sys.eps_r * std::pow(sys.c_hbarc, 3), 2);
    }

    double get_E_min() const {
        return -0.25 * param_c6 * param_c6 / param_c12;
    }

    double get_r_min() const {
        return 0.8 /
               std::pow(0.5 * param_c6 / param_c12 * (1 + M_SQRT2), 1.0 / 6.0);
    }

    double operator()(double x) const {
        return param_c12 / std::pow(x, 12) - param_c6 / std::pow(x, 6);
    };
};

double biexciton_lj_c6(double be_cou, const system_data& sys) {
    return biexciton_pot_lj_s::get_c6(be_cou, sys);
}

std::vector<double> biexciton_pot_lj_vec(
    double param_c12,
    double be_cou,
    double r_max,
    uint32_t n_steps,
    const system_data& sys) {
    biexciton_pot_lj_s pot{
        param_c12,
        biexciton_pot_lj_s::get_c6(be_cou, sys),
        sys,
    };

    std::vector<double> r(n_steps);
    double r_min{pot.get_r_min()};

#pragma omp parallel for
    for (uint32_t i = 0; i < n_steps; i++) {
        r[i] = pot(r_min + (r_max - r_min) / (n_steps - 1) * i);
    }

    return r;
}

std::vector<double> biexciton_wf_lj(
    double be_biexc,
    double be_cou,
    double param_c12,
    double r_max,
    uint32_t n_steps,
    const system_data& sys) {
    biexciton_pot_lj_s pot{
        param_c12,
        biexciton_pot_lj_s::get_c6(be_cou, sys),
        sys,
    };

    auto [f_vec, t_vec] = wf_gen_s_r_t<biexciton_pot_lj_s>(
        be_biexc, pot.get_r_min(), sys.c_alpha_bexc, r_max, n_steps, pot);

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double biexciton_be_lj(
    double param_c12, double be_cou, const system_data& sys) {
    biexciton_pot_lj_s pot{
        param_c12,
        biexciton_pot_lj_s::get_c6(be_cou, sys),
        sys,
    };

    double E_min{pot.get_E_min()};
    double r_min{pot.get_r_min()};

    return wf_gen_E_t<biexciton_pot_lj_s>(E_min, r_min, sys.c_alpha_bexc, pot);
}

std::vector<double> biexciton_be_lj_vec(
    const std::vector<double>& c12_vec,
    double be_cou,
    const system_data& sys) {
    std::vector<double> r(c12_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < c12_vec.size(); i++) {
        r[i] = biexciton_be_lj(c12_vec[i], be_cou, sys);
    }

    return r;
}

template <class pot_s>
double biexciton_c12_f(double param_c12, biexciton_c12_s<pot_s>* s) {
    s->pot.param_c12 = param_c12;

    double r{
        wf_gen_E_t<pot_s>(
            s->pot.get_E_min(), s->pot.get_r_min(), s->sys.c_alpha_bexc,
            s->pot),
    };

    if (std::isnan(r)) {
        return s->pot.get_E_min();
    }

    return r - s->be_biexc;
}

double biexciton_c12_lj(
    double be_cou, double be_biexc, const system_data& sys) {
    using pot_s = biexciton_pot_lj_s;
    constexpr uint32_t local_max_iter{1 << 7};

    double z_min{1e-3}, z_max{1e3}, z;

    pot_s pot{
        z_min,
        biexciton_pot_lj_s::get_c6(be_cou, sys),
        sys,
    };

    biexciton_c12_s<pot_s> params{be_biexc, pot, sys};

    gsl_function funct;
    funct.function =
        &templated_f<biexciton_c12_s<pot_s>, biexciton_c12_f<pot_s>>;
    funct.params = &params;

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* s            = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < local_max_iter; iter++) {
        status = gsl_root_fsolver_iterate(s);
        z      = gsl_root_fsolver_root(s);
        z_min  = gsl_root_fsolver_x_lower(s);
        z_max  = gsl_root_fsolver_x_upper(s);

        // status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
        status =
            gsl_root_test_residual(funct.function(z, &params), global_eps);

        /*
        printf(
            "[%d] iterating -- (%f, %f), (%.2e, %.2e)\n", iter, z_min, z_max,
            funct.function(z_min, &params), funct.function(z_max, &params));
        */
    }

    gsl_root_fsolver_free(s);
    return z;
}
