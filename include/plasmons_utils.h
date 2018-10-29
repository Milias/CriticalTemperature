#pragma once
#include "common.h"

struct plasmon_kmax_s {
    double mu_e;
    double mu_h;
    double v_1;

    const system_data & sys;
};

struct plasmon_disp_s {
    double k;
    double mu_e;
    double mu_h;
    double v_1;

    const system_data & sys;
};

struct plasmon_disp_inv_s {
    double w;
    double mu_e;
    double mu_h;
    double v_1;

    const system_data & sys;
};

struct plasmon_potcoef_s {
    double w;
    double k1;
    double k2;
    double mu_e;
    double mu_h;
    double v_1;

    const system_data & sys;

    double delta{1e-2};
};

struct plasmon_potcoef_lwl_s {
    double k1;
    double k2;
    double ls;
    double v_1;

    const system_data & sys;
};

struct plasmon_sysmat_det_zero_s {
    const arma::cx_mat & mat_elem;
    const arma::cx_mat & mat_kron;
    const arma::cx_mat & mat_G0;

    arma::cx_mat & mat_z_G0;
    arma::cx_mat & mat_potcoef;
};

struct plasmon_sysmat_det_zero_lwl_s {
    const arma::mat & mat_elem;
    const arma::mat & mat_kron;
    const arma::mat & mat_G0;

    arma::mat & mat_z_G0;
    arma::mat & mat_potcoef;
};

struct plasmon_real_potcoef_k_s {
    double x;
    double mu_e;
    double mu_h;
    double v_1;

    const system_data & sys;
};
