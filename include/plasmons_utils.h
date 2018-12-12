#pragma once
#include "common.h"

struct plasmon_potcoef_s {
    double w;
    double k1;
    double k2;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

template <typename T>
struct plasmon_mat_s {
    using type = T;

    arma::Mat<T>& mat_elem;
    const arma::Row<T>& mat_G0;

    arma::Row<T>& mat_z_G0;
    arma::Mat<T>& mat_potcoef;

    double delta{1e-12};
};

struct plasmon_rpot_s {
    double x;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_exc_mu_zero_s {
    const system_data& sys;

    double val{0.0};
};

template <typename T>
struct plasmon_exc_mu_lim_s {
    uint32_t N_k;
    uint32_t N_w;

    const system_data& sys;

    plasmon_mat_s<T>& mat_s;

    double val;
    double eb_lim{std::numeric_limits<double>::quiet_NaN()};
    double delta{1e-12};
};

template <typename T>
struct plasmon_density_s {
    double n_total;

    uint32_t N_k;
    uint32_t N_w;

    double mu_e_lim;
    double eb_lim;
    const system_data& sys;

    plasmon_mat_s<T>& mat_s;

    double delta{1e-12};
};
