#include "wavefunction_bexc.h"

std::vector<double> wf_bexc_s(
    double E, double lj_param_a, double lj_param_b, const system_data& sys) {
    auto [f_vec, t_vec] = wf_bexc_s_t<true, 0>(E, lj_param_a, lj_param_b, sys);
    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

std::vector<double> wf_bexc_s_r(
    double rmax,
    double E,
    double lj_param_a,
    double lj_param_b,
    const system_data& sys) {
    auto [f_vec, t_vec] = wf_bexc_s_r_t<0>(rmax, E, lj_param_a, lj_param_b, sys);
    std::vector<double> r(3 * f_vec.size());

    for (uint32_t i = 0; i < f_vec.size(); i++) {
        r[3 * i]     = f_vec[i][0];
        r[3 * i + 1] = f_vec[i][1];
        r[3 * i + 2] = t_vec[i];
    }

    return r;
}

uint32_t wf_bexc_n(
    double E, double lj_param_a, double lj_param_b, const system_data& sys) {
    return wf_bexc_n_t<0>(E, lj_param_a, lj_param_b, sys);
}

double wf_bexc_E(
    double E_min,
    double lj_param_a,
    double lj_param_b,
    const system_data& sys) {
    return wf_bexc_E_t<0>(E_min, lj_param_a, lj_param_b, sys);
}

