#pragma once
#include "common.h"

struct plasmon_kmax_s {
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_disp_s {
    double k;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_disp_inv_s {
    double w;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_disp_th_s {
    double w;
    double k0;
    double k1;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_potcoef_s {
    double w;
    double k1;
    double k2;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_potcoef_lwl_s {
    double k1;
    double k2;
    double ls;

    const system_data& sys;

    double delta{1e-12};
};

template <typename T>
struct plasmon_mat_s {
    using type = T;

    const arma::Mat<T>& mat_elem;
    const arma::SpMat<T>& mat_kron;
    const arma::SpMat<T>& mat_G0;

    arma::SpMat<T>& mat_z_G0;
    arma::Mat<T>& mat_potcoef;

    double delta{1e-12};
};

struct plasmon_real_potcoef_k_s {
    double x;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};
