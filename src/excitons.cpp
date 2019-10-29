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

    auto [f_vec, t_vec] = wf_gen_s_r_t<exciton_pot_cou_s>(
        be_exc, 0.0, sys.c_alpha, r_max, n_steps, pot);

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

struct exciton_pot_ke_s {
    double size_d;
    double eps;

    const system_data& sys;

    double operator()(double r) const {
        double x{sys.eps_r / eps * r / size_d};
        double pot{struve(0.0, x) - gsl_sf_bessel_Y0(x)};
        return -sys.c_aEM / eps * sys.c_hbarc / size_d * 0.5 * M_PI * pot;
    };
};

std::vector<double> exciton_pot_ke_vec(
    double size_d,
    double eps,
    const std::vector<double>& x_vec,
    const system_data& sys) {
    exciton_pot_ke_s pot{
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

std::vector<double> exciton_wf_ke(
    double be_exc,
    double size_d,
    double eps,
    double r_max,
    uint32_t n_steps,
    const system_data& sys) {
    exciton_pot_ke_s pot{
        size_d,
        eps,
        sys,
    };

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

double exciton_be_ke(
    double size_d, double eps, const system_data& sys) {
    exciton_pot_ke_s pot{
        size_d,
        eps,
        sys,
    };

    return wf_gen_E_t<exciton_pot_ke_s>(
        2 * sys.get_E_n(0.5), 0.0, sys.c_alpha, pot);
}
