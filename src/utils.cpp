#include "utils.h"

struct exc_mu_s {
    const system_data& sys;

    double val{0.0};
};

double exc_mu_zero_f(double x, exc_mu_s* s) {
    return x - std::pow(1 + x, 1 - s->sys.m_eh);
}

double exc_mu_val_f(double x, exc_mu_s* s) {
    return x + s->sys.get_mu_h(x) - s->val;
}

double exc_mu_val_df(double x, exc_mu_s* s) {
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

system_data::system_data(double m_e, double m_h, double eps_r, double T) :
    m_e(m_e * c_m_e),
    m_h(m_h * c_m_e),
    eps_r(eps_r),
    dl_m_e(m_e),
    dl_m_h(m_h),
    m_p(c_m_e * m_e * m_h / (m_e + m_h)),
    m_2p(2.0 * m_p),
    m_pe(m_h / (m_e + m_h)),
    m_ph(m_e / (m_e + m_h)),
    m_sigma((m_e + m_h) * (1.0 / m_e + 1.0 / m_h)),
    m_eh(m_e / m_h),
    c_alpha(0.5 * c_hbarc * c_hbarc / m_p),
    c_alpha_bexc(c_hbarc * c_hbarc / (m_e + m_h) / c_m_e),
    a0(eps_r / c_aEM * c_hbarc / m_p),
    T(T),
    beta(f_beta(T)),
    energy_th(f_energy_th(T)),
    // sys_ls(M_1_PI * c_aEM / eps_r * (m_e + m_h) / c_hbarc),
    sys_ls(M_1_PI * c_aEM / eps_r / c_hbarc * m_p * (1 / m_pe + 1 / m_ph)),
    zt_len(0.5 * c_hbarc / m_2p) {
    lambda_th       = f_lambda_th(beta, m_p);
    lambda_th_biexc = 2 * c_hbarc * std::sqrt(M_PI * beta / (m_e + m_h) / c_m_e);
    m_pT            = m_p / energy_th;
    E_1             = -0.5 * m_p * std::pow(c_aEM / eps_r, 2);
    delta_E         = std::pow(2, 1.75) * c_aEM / eps_r *
              std::sqrt(m_pT * M_PI * c_aEM / eps_r * std::sqrt(m_pT));
}

void system_data::set_temperature(double T) {
    this->T   = T;
    beta      = f_beta(T);
    energy_th = f_energy_th(T);
    lambda_th = f_lambda_th(beta, m_p);
    m_pT      = m_p / energy_th;
}

double system_data::get_z1(double E, double mu_t) const {
    return E / m_sigma - mu_t;
    // return 0.25 * (1 - std::pow(m_ph - m_pe, 2)) * E - mu_t;
}

double system_data::get_E_n(double n) const {
    return E_1 / (n * n);
}

double system_data::get_mu_h_t0(double mu_e) const {
    return m_eh * mu_e;
}

double system_data::get_mu_h_ht(double mu_e) const {
    return mu_e + std::log(m_eh) / beta;
}

double system_data::get_mu_h(double mu_e) const {
    double result{
        std::log(std::pow(1 + std::exp(beta * mu_e), m_eh) - 1) / beta,
    };

    if (std::isinf(result)) {
        if (mu_e > 0) {
            return get_mu_h_t0(mu_e);
        } else {
            return get_mu_h_ht(mu_e);
        }
    } else {
        return result;
    }
}

double system_data::density_ideal_t0(double mu_e) const {
    if (mu_e <= 0) {
        return 0.0;
    }

    return m_e * M_1_PI * mu_e / std::pow(c_hbarc, 2);
}

double system_data::density_ideal_ht(double mu_e) const {
    return m_e * M_1_PI * std::exp(beta * mu_e) /
           (std::pow(c_hbarc, 2) * beta);
}

double system_data::density_ideal(double mu_e) const {
    double result{
        m_e * M_1_PI * logExp(beta * mu_e) / (std::pow(c_hbarc, 2) * beta),
    };

    return result;
}

double system_data::density_exc_ht(double mu_ex, double eb) const {
    return 2 * (m_e + m_h) * M_1_PI * std::exp(beta * (mu_ex - eb)) /
           (std::pow(c_hbarc, 2) * beta);
}

double system_data::density_exc_exp(double u) const {
    double log_val;

    if (u > -10) {
        log_val = std::log(1.0 - 1.0 / (1.0 + std::exp(u)));
    } else if (u > -50) {
        mpfr_t y;
        mpfr_init_set_d(y, u, MPFR_RNDN);
        mpfr_exp(y, y, MPFR_RNDN);
        mpfr_add_ui(y, y, 1, MPFR_RNDN);
        mpfr_ui_div(y, 1, y, MPFR_RNDN);
        mpfr_ui_sub(y, 1, y, MPFR_RNDN);
        mpfr_log(y, y, MPFR_RNDN);
        log_val = mpfr_get_d(y, MPFR_RNDN);

        mpfr_clear(y);
    } else {
        log_val = u;
    }

    double result{
        -2 * (m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) * log_val,
    };

    return result;
}

double system_data::density_exc_exp_ht(double u) const {
    double log_val;

    if (u > -10) {
        log_val = 1.0 + std::exp(u);
    } else if (u > -50) {
        mpfr_t y;
        mpfr_init_set_d(y, u, MPFR_RNDN);
        mpfr_exp(y, y, MPFR_RNDN);
        mpfr_add_ui(y, y, 1, MPFR_RNDN);
        log_val = mpfr_get_d(y, MPFR_RNDN);

        mpfr_clear(y);
    } else {
        return density_exc_exp(u);
    }

    double result{
        2 * (m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) / log_val,
    };

    return result;
}

double system_data::density_exc(double mu_ex, double eb) const {
    const double r{
        -2 * (m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) *
            std::log(1 - std::exp(beta * (mu_ex - eb))),
    };

    if (r == 0.0) {
        return density_exc_ht(mu_ex, eb);
    }

    return r;
}

double system_data::density_exc2(
    double mu_ex, double eb_ex, double eb_ex2) const {
    double const r{
        -(m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) *
            std::log(1 - std::exp(beta * (2 * mu_ex - 2 * eb_ex - eb_ex2))),
    };

    if (std::isnan(r)) {
        mpfr_t y;
        /* mpfr_init_set_d(y, beta * (2 * mu_ex - 2 * eb_ex - eb_ex2),
         * MPFR_RNDN); */
        mpfr_init_set_d(y, 2 * mu_ex, MPFR_RNDN);
        mpfr_sub_d(y, y, 2 * eb_ex, MPFR_RNDN);
        mpfr_sub_d(y, y, eb_ex2, MPFR_RNDN);
        mpfr_mul_d(y, y, beta, MPFR_RNDN);
        // mpfr_exp(y, y, MPFR_RNDN);
        // mpfr_ui_sub(y, 1, y, MPFR_RNDN);
        mpfr_neg(y, y, MPFR_RNDN);
        mpfr_log(y, y, MPFR_RNDN);
        double log_val = mpfr_get_d(y, MPFR_RNDN);

        mpfr_clear(y);
        return -(m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) * log_val;
    }

    return r;
}

double system_data::density_exc2_u(double u) const {
    /*
     * beta * (2 * mu_ex - 2 * eb_ex - eb_ex2) = - log(1 + exp(u))
     */

    if (u > 20) {
        return (m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) *
               std::exp(-u);
    } else if (u < -20) {
        return -(m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) * u;
    }

    double const r{
        -(m_e + m_h) * M_1_PI / (std::pow(c_hbarc, 2) * beta) *
            std::log(1.0 - 1.0 / (1.0 + std::exp(u))),
    };

    return r;
}

double system_data::mu_ideal(double n) const {
    const double r{
        std::log(std::exp(M_PI * c_hbarc * c_hbarc / m_e * beta * n) - 1) /
            beta,
    };

    if (std::isinf(r)) {
        return M_PI * c_hbarc * c_hbarc / m_e * n;
    } else if (r < 1e-10) {
        return std::log(M_PI * c_hbarc * c_hbarc / m_e * beta * n) / beta +
               0.5 * M_PI * c_hbarc * c_hbarc / m_e * n;
    }

    return r;
}

double system_data::mu_h_ideal(double n) const {
    const double r{
        std::log(std::exp(M_PI * c_hbarc * c_hbarc / m_h * beta * n) - 1) /
            beta,
    };

    if (std::isinf(r)) {
        return M_PI * c_hbarc * c_hbarc / m_h * n;
    }

    return r;
}

double system_data::mu_exc_u(double n) const {
    const double r{std::log(
        1.0 / (1.0 -
               std::exp(
                   -0.5 * M_PI * c_hbarc * c_hbarc / (m_e + m_h) * beta * n)) -
        1.0)};

    if (std::isinf(r)) {
        if (r < 0) {
            return -beta * 0.5 * M_PI * c_hbarc * c_hbarc * n / (m_e + m_h);
        } else {
            return std::log(
                (m_e + m_h) / (0.5 * M_PI * c_hbarc * c_hbarc * beta * n));
        }
    }

    return r;
}

double system_data::ls_ideal(double n) const {
    return M_1_PI * c_aEM / (eps_r * c_hbarc) * m_p *
           ((1 - std::exp(-M_PI * c_hbarc * c_hbarc / m_p * m_pe * beta * n)) /
                m_pe +
            (1 - std::exp(-M_PI * c_hbarc * c_hbarc / m_p * m_ph * beta * n)) /
                m_ph);
}

double system_data::exc_mu_zero() const {
    exc_mu_s params{*this};

    gsl_function funct;
    funct.function = &templated_f<exc_mu_s, exc_mu_zero_f>;
    funct.params   = &params;

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
            return std::log(z_max) / beta;

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

    return std::log(z) / beta;
}

double system_data::exc_mu_val(double val) const {
    if (val == 0) {
        return exc_mu_zero();
    }

    constexpr uint32_t local_max_iter{1 << 7};

    exc_mu_s s{*this, val};

    gsl_function_fdf funct;
    funct.f      = &templated_f<exc_mu_s, exc_mu_val_f>;
    funct.df     = &templated_f<exc_mu_s, exc_mu_val_df>;
    funct.fdf    = &templated_fdf<exc_mu_s, exc_mu_val_f, exc_mu_val_df>;
    funct.params = &s;

    double z{
        std::max(0.5 * (val - std::log(m_eh) / beta), val / (1 + m_eh)),
    };
    double z0{0};

    if (std::abs(funct.f(z, &s)) < 1e-8) {
        return z;
    }

    const gsl_root_fdfsolver_type* solver_type = gsl_root_fdfsolver_steffenson;
    gsl_root_fdfsolver* solver = gsl_root_fdfsolver_alloc(solver_type);

    gsl_root_fdfsolver_set(solver, &funct, z);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < local_max_iter; iter++) {
        z0     = z;
        status = gsl_root_fdfsolver_iterate(solver);
        z      = gsl_root_fdfsolver_root(solver);
        status = gsl_root_test_residual(funct.f(z, &s), 1e-8);
    }

    gsl_root_fdfsolver_free(solver);

    return z;
}
