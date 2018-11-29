#pragma once
#include "common.h"

struct plasmon_kmax_s {
    double mu_e;
    double mu_h;

    const system_data& sys;
};

struct plasmon_disp_s {
    double k;
    double mu_e;
    double mu_h;

    const system_data& sys;
};

struct plasmon_disp_inv_s {
    double w;
    double mu_e;
    double mu_h;

    const system_data& sys;
};

struct plasmon_disp_th_s {
    double w;
    double k0;
    double k1;
    double mu_e;
    double mu_h;

    const system_data& sys;
};

struct plasmon_potcoef_s {
    double w;
    double k1;
    double k2;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-2};
};

struct plasmon_potcoef_lwl_s {
    double k1;
    double k2;
    double ls;

    const system_data& sys;
};

struct plasmon_det_s {
    const arma::cx_mat& mat_elem;
    const arma::sp_cx_mat& mat_kron;
    const arma::sp_cx_mat& mat_G0;

    arma::sp_cx_mat& mat_z_G0;
    arma::cx_mat& mat_potcoef;
};

struct plasmon_det_rs {
    const arma::mat& mat_elem;
    const arma::sp_mat& mat_kron;
    const arma::sp_mat& mat_G0;

    arma::sp_mat& mat_z_G0;
    arma::mat& mat_potcoef;
};

struct plasmon_real_potcoef_k_s {
    double x;
    double mu_e;
    double mu_h;

    const system_data& sys;
};
