#include "utils.h"

system_data::system_data(double m_e, double m_h, double eps_r, double T) :
    m_e(m_e * c_m_e),
    m_h(m_h * c_m_e),
    eps_r(eps_r),
    a0(eps_r * c_hbarc * (m_e + m_h) /
       (c_aEM * c_m_e * m_e * m_h)), // eps_r / c_aEM * c_hbarc / m_p;
    dl_m_e(m_e),
    dl_m_h(m_h),
    m_p(c_m_e * m_e * m_h / (m_e + m_h)),
    m_2p(2.0 * m_p),
    m_pe(m_h / (m_e + m_h)),
    m_ph(m_e / (m_e + m_h)),
    m_sigma((m_e + m_h) * (1.0 / m_e + 1.0 / m_h)),
    m_eh(m_e / m_h),
    c_alpha(0.5 * c_hbarc * c_hbarc / m_p),
    T(T),
    beta(f_beta(T)),
    energy_th(f_energy_th(T)),
    zt_len(0.5 * c_hbarc / m_2p) {
    lambda_th = f_lambda_th(beta, m_p);
    m_pT      = m_p / energy_th;
    E_1       = -0.5 * m_p * std::pow(c_aEM / eps_r, 2);
    delta_E   = std::pow(2, 1.75) * c_aEM / eps_r *
              std::sqrt(m_pT * M_PI * c_aEM / eps_r * std::sqrt(m_pT));
    sys_ls = 0.5 * M_1_PI * c_aEM / eps_r * (this->m_e + this->m_h) / c_hbarc;
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

double system_data::mu_ideal(double n) const {
    const double r{
        std::log(std::exp(M_PI * c_hbarc * c_hbarc / m_e * beta * n) - 1),
    };

    if (std::isinf(r)) {
        return M_PI * c_hbarc * c_hbarc / m_e * beta * n;
    }

    return r;
}

double system_data::mu_h_ideal(double n) const {
    const double r{
        std::log(std::exp(M_PI * c_hbarc * c_hbarc / m_h * beta * n) - 1),
    };

    if (std::isinf(r)) {
        return M_PI * c_hbarc * c_hbarc / m_h * beta * n;
    }

    return r;
}

double system_data::mu_exc_u(double n) const {
    const double r{
        std::log(
            1 / (1 - std::exp(
                         -0.5 * M_PI * c_hbarc * c_hbarc / (m_e + m_h) * beta *
                         n)) -
            1),
    };

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
