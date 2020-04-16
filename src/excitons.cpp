#include "excitons.h"

struct exciton_pot_cou_s {
    const system_data& sys;

    double operator()(double r) const {
        return -sys.c_aEM / sys.eps_r * sys.c_hbarc / r;
    };
};

std::vector<double> exciton_pot_cou_vec(
    const std::vector<double>& x_vec, const system_data& sys) {
    exciton_pot_cou_s pot{sys};

    std::vector<double> r(x_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_vec.size(); i++) {
        r[i] = pot(x_vec[i]);
    }

    return r;
}

std::vector<double> exciton_wf_cou(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys) {
    exciton_pot_cou_s pot{sys};

    ///*
    auto [f_vec, t_vec] = wf_gen_s_r_t<exciton_pot_cou_s>(
        be_exc, 0.0, sys.c_alpha, r_max, n_steps, pot);
    //*/

    /*
    auto [f_vec, t_vec] = wf_gen_s_t<true, exciton_pot_cou_s>(
        be_exc, 0.0, sys.c_alpha, pot, r_max);
    */

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double exciton_be_cou(const system_data& sys) {
    exciton_pot_cou_s pot{sys};

    return wf_gen_E_t<exciton_pot_cou_s>(
        1.1 * sys.get_E_n(0.5), 0.0, sys.c_alpha, pot);
}

struct exciton_pot_hn_s {
    double size_d;
    double eps;

    const system_data& sys;

    double operator()(double r) const {
        double x{2 * sys.eps_r / eps * r / size_d};
        double pot{struve(0, x) - gsl_sf_bessel_Y0(x)};
        return -sys.c_aEM / eps * sys.c_hbarc / size_d * M_PI * pot;
    };
};

std::vector<double> exciton_pot_hn_vec(
    double size_d,
    double eps,
    const std::vector<double>& x_vec,
    const system_data& sys) {
    exciton_pot_hn_s pot{
        size_d,
        eps,
        sys,
    };

    std::vector<double> r(x_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_vec.size(); i++) {
        r[i] = pot(x_vec[i]);
    }

    return r;
}

std::vector<double> exciton_wf_hn(
    double be_exc,
    double size_d,
    double eps,
    double r_max,
    uint32_t n_steps,
    const system_data& sys) {
    exciton_pot_hn_s pot{
        size_d,
        eps,
        sys,
    };

    auto [f_vec, t_vec] = wf_gen_s_r_t<exciton_pot_hn_s>(
        be_exc, 0.0, sys.c_alpha, r_max, n_steps, pot);

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double exciton_be_hn(double size_d, double eps, const system_data& sys) {
    exciton_pot_hn_s pot{
        size_d,
        eps,
        sys,
    };

    return wf_gen_E_t<exciton_pot_hn_s>(
        1.1 * sys.get_E_n(0.5), 0.0, sys.c_alpha, pot);
}

struct exciton_pot_hnlr_s {
    const system_data& sys;

    double operator()(double r) const {
        double x{0.5 * sys.eps_mat / sys.eps_r / r * sys.size_d};
        double pot{x - std::pow(x, 3) + 9 * std::pow(x, 5)};
        return -sys.c_aEM * sys.c_hbarc / sys.eps_mat / sys.size_d * 2 * pot;
    };
};

std::vector<double> exciton_pot_hnlr_vec(
    const std::vector<double>& x_vec, const system_data& sys) {
    exciton_pot_hnlr_s pot{sys};

    std::vector<double> r(x_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_vec.size(); i++) {
        r[i] = pot(x_vec[i]);
    }

    return r;
}

std::vector<double> exciton_wf_hnlr(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys) {
    exciton_pot_hnlr_s pot{sys};

    auto [f_vec, t_vec] = wf_gen_s_r_t<exciton_pot_hnlr_s>(
        be_exc, 0.0, sys.c_alpha, r_max, n_steps, pot);

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double exciton_be_hnlr(const system_data& sys) {
    exciton_pot_hnlr_s pot{sys};

    return wf_gen_E_t<exciton_pot_hnlr_s>(
        1.1 * sys.get_E_n(0.5), 0.0, sys.c_alpha, pot);
}

struct ke_k_pot_s {
    double r;
    const system_data& sys;
};

template <bool bessel = true>
double ke_k_pot(double u, ke_k_pot_s* params) {
    double pot{1};

    if constexpr (bessel) {
        pot *= gsl_sf_bessel_J0(u);
    } else {
        u += 0.25 * M_PI;
        pot /= std::sqrt(0.5 * M_PI * u);
    }

    if (std::abs(params->sys.eps_mat - params->sys.eps_r) >= 1e-8) {
        pot /= std::tanh(
            u * params->sys.size_d * 0.5 / params->r + params->sys.eta_func());
    }

    return pot;
}

struct exciton_pot_ke_s {
    const system_data& sys;

    double operator()(double r) const {
        constexpr uint32_t n_int{2};
        constexpr uint32_t local_ws_size{1 << 4};
        constexpr uint32_t t_size{1 << 4};

        double result[n_int] = {0}, error[n_int] = {0};
        constexpr uint32_t n_sum{1 << 4};

        double result_sum = {0.0};
        double last_zero  = {0.0};
        double t[n_sum];

        ke_k_pot_s s{r, sys};

        gsl_function integrands[n_int];

        integrands[0].function = &templated_f<ke_k_pot_s, ke_k_pot<true>>;
        integrands[0].params   = &s;

        integrands[1].function = &templated_f<ke_k_pot_s, ke_k_pot<false>>;
        integrands[1].params   = &s;

        gsl_integration_qawo_table* qawo_table[2] = {
            gsl_integration_qawo_table_alloc(r, 1, GSL_INTEG_COSINE, t_size),
            gsl_integration_qawo_table_alloc(r, 1, GSL_INTEG_SINE, t_size),
        };

        gsl_integration_workspace* ws[2] = {
            gsl_integration_workspace_alloc(local_ws_size),
            gsl_integration_workspace_alloc(local_ws_size),
        };

        for (uint32_t i = 0; i < n_sum; i++) {
            double temp{gsl_sf_bessel_zero_J0(i + 1)};

            gsl_integration_qag(
                integrands, last_zero, temp, 0.0, global_eps, local_ws_size,
                GSL_INTEG_GAUSS15, ws[0], result, error);

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

        return -sys.c_aEM * sys.c_hbarc / r / sys.eps_mat * result_sum;
    };
};

std::vector<double> exciton_pot_ke_vec(
    const std::vector<double>& x_vec, const system_data& sys) {
    exciton_pot_ke_s pot{sys};

    std::vector<double> r(x_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < x_vec.size(); i++) {
        r[i] = pot(x_vec[i]);
    }

    return r;
}

std::vector<double> exciton_wf_ke(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys) {
    exciton_pot_ke_s pot{sys};

    auto [f_vec, t_vec] = wf_gen_s_r_t<exciton_pot_ke_s>(
        be_exc, 0.0, sys.c_alpha, r_max, n_steps, pot);

    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

double exciton_be_ke(const system_data& sys) {
    exciton_pot_ke_s pot{sys};

    return wf_gen_E_t<exciton_pot_ke_s>(
        1.1 * sys.get_E_n(0.5), 0.0, sys.c_alpha, pot);
}

double exciton_os(double Lx, double Ly, uint32_t nx, uint32_t ny) {
    return Lx * Ly / (nx * ny);
}

double exciton_cm_se(
    double Lx, double Ly, uint32_t nx, uint32_t ny, const system_data& sys) {
    return (std::pow(nx / Lx, 2) + std::pow(ny / Ly, 2)) *
           std::pow(M_PI * sys.c_hbarc, 2) * 0.5 / (sys.m_e + sys.m_h);
}

double exciton_cm_sep(
    double th, uint32_t nx, uint32_t ny, const system_data& sys) {
    return 0.5 * std::pow(M_PI * sys.c_hbarc, 2) / (sys.m_e + sys.m_p) *
           (std::pow(nx / sys.sigma_x / std::cos(th), 2) +
            std::pow(ny / sys.sigma_y / std::sin(th), 2));
}

double exciton_PL_dirac_f(double th, exciton_PL_s* s) {
    if (th == 0 || th == 0.5 * M_PI) {
        return 0;
    }

    const double E_S{exciton_cm_sep(th, s->nx, s->ny, s->sys)};

    const double r{
        -0.5 * E_S / s->energy +
            std::sqrt(E_S / s->energy) *
                (s->sys.size_Lx / s->sys.sigma_x * std::cos(th) +
                 s->sys.size_Ly / s->sys.sigma_y * std::sin(th)),
    };

    return std::cos(th) * std::sin(th) * E_S * E_S * std::exp(r);
}

double exciton_PL_d(
    double energy, uint32_t nx, uint32_t ny, const system_data& sys) {
    if (energy <= 0) {
        return 0;
    }

    constexpr uint32_t n_int{1};
    constexpr uint32_t local_ws_size{1 << 7};

    double result[n_int] = {0}, error[n_int] = {0};

    exciton_PL_s s{energy, nx, ny, sys};

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<exciton_PL_s, exciton_PL_dirac_f>;
    integrands[0].params   = &s;

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, 0.5 * M_PI, 0, global_eps, local_ws_size,
        GSL_INTEG_GAUSS31, ws, result, error);

    gsl_integration_workspace_free(ws);

    return M_1_PI / std::pow(energy * nx * ny, 2) *
           std::exp(
               -0.5 * (std::pow(sys.size_Lx / sys.sigma_x, 2) +
                       std::pow(sys.size_Ly / sys.sigma_y, 2)) -
               sys.beta * energy) *
           result[0];
}

std::vector<double> exciton_PL_d_vec(
    const std::vector<double>& energy_vec,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys) {
    std::vector<double> r(energy_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < energy_vec.size(); i++) {
        r[i] = exciton_PL_d(energy_vec[i], nx, ny, sys);
    }

    return r;
}

double exciton_PL_f_th(double th, exciton_PL_s* s) {
    if (th == 0 || th == 0.5 * M_PI) {
        return 0;
    }

    const double E_S{
        exciton_cm_sep(th, s->nx, s->ny, s->sys) / std::pow(s->t, 2),
    };

    const double r{
        -0.5 * s->t * s->t +
            s->t * (s->sys.size_Lx / s->sys.sigma_x * std::cos(th) +
                    s->sys.size_Ly / s->sys.sigma_y * std::sin(th)) -
            s->sys.beta * E_S,
    };

    return std::cos(th) * std::sin(th) * std::exp(r) *
           gsl_ran_cauchy_pdf(s->energy - E_S, 0.5 * s->gamma);
}

double exciton_PL_f_t(double t, exciton_PL_s* s) {
    if (t == 0) {
        return 0;
    }

    s->t = t;

    constexpr uint32_t n_int{1};
    constexpr uint32_t local_ws_size{1 << 6};

    double result[n_int] = {0}, error[n_int] = {0};

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<exciton_PL_s, exciton_PL_f_th>;
    integrands[0].params   = s;

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, 0, 0.5 * M_PI, 0, global_eps, local_ws_size,
        GSL_INTEG_GAUSS31, ws, result, error);

    gsl_integration_workspace_free(ws);

    return std::pow(t, 3) * result[0];
}

double exciton_PL(
    double energy,
    double gamma,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys) {
    constexpr uint32_t n_int{1};
    constexpr uint32_t local_ws_size{1 << 6};

    double result[n_int] = {0}, error[n_int] = {0};

    exciton_PL_s s{energy, nx, ny, sys, gamma};

    gsl_function integrands[n_int];

    integrands[0].function = &templated_f<exciton_PL_s, exciton_PL_f_t>;
    integrands[0].params   = &s;

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qagiu(
        integrands, 0, 0, global_eps, local_ws_size, ws, result, error);

    gsl_integration_workspace_free(ws);

    return 0.5 * M_1_PI / std::pow(nx * ny, 2) *
           std::exp(
               -0.5 * (std::pow(sys.size_Lx / sys.sigma_x, 2) +
                       std::pow(sys.size_Ly / sys.sigma_y, 2))) *
           result[0];
}

std::vector<double> exciton_PL_vec(
    const std::vector<double>& energy_vec,
    double gamma,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys) {
    std::vector<double> r(energy_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < energy_vec.size(); i++) {
        r[i] = exciton_PL(energy_vec[i], gamma, nx, ny, sys);
    }

    return r;
}

double exciton_dd_var_E(
    double alpha,
    double rho_cm,
    double th_cm,
    double phi,
    double delta_z,
    const system_data& sys) {
    return (sys.get_E_n(0.5) -
            9 * alpha * std::sqrt(3 - 3 * alpha * alpha) *
                (16 * alpha * alpha - 7) * std::sin(th_cm + phi) / 32 *
                std::pow(sys.c_hbarc, 2) *
                (0.5 * sys.size_d + sys.ext_dist_l) * delta_z * rho_cm /
                sys.m_p /
                std::pow(
                    std::pow(0.5 * sys.size_d + sys.ext_dist_l, 2) +
                        rho_cm * rho_cm,
                    2.5)) /
           (1 + 8 * alpha * alpha);
}

double exciton_dd_var_E_a0(
    double alpha,
    double a0,
    double rho_cm,
    double th_cm,
    double phi,
    double delta_z,
    const system_data& sys) {
    return -sys.c_aEM * sys.c_hbarc / sys.eps_mat * 2 *
           (1 - sys.c_hbarc * sys.eps_mat / (4 * sys.m_p * a0 * sys.c_aEM) +
            9 * alpha * std::sqrt(3 - 3 * alpha * alpha) *
                (16 * alpha * alpha - 7) * std::sin(th_cm + phi) / 32 * a0 *
                a0 * (0.5 * sys.size_d + sys.ext_dist_l) * delta_z * rho_cm /
                std::pow(
                    std::pow(0.5 * sys.size_d + sys.ext_dist_l, 2) +
                        rho_cm * rho_cm,
                    2.5)) /
           a0 / (1 + 8 * alpha * alpha);
}
