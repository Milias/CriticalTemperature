#pragma once
#include "common.h"
#include "templates.h"

struct plasmon_elem_s {
    /*
     * ids[4] = {i, j, k, l}
     * wkwk[4] = {w', k', w'', k''}
     */
    std::vector<uint32_t> id;
    std::vector<double> wkwk;

    std::complex<double> val{0};

    plasmon_elem_s() = default;

    plasmon_elem_s(
        const std::vector<uint32_t> id,
        const std::vector<double> wkwk,
        const std::complex<double> val = {0}
    ) :
        id(id),
        wkwk(wkwk),
        val(val)
    {}
};

std::vector<double> plasmon_green(double w, double k, double mu_e, double mu_h,
    double v_1, const system_data & sys, double delta = 1e-6);

double plasmon_kmax(double mu_e, double mu_h, double v_1,
    const system_data & sys);
double plasmon_wmax(double mu_e, double mu_h, double v_1,
    const system_data & sys);
double plasmon_wmax(double kmax, double mu_e, const system_data & sys);

double plasmon_disp(double k, double mu_e, double mu_h, double v_1,
    const system_data & sys);
double plasmon_disp_ncb(double k, double mu_e, double mu_h, double v_1,
    const system_data & sys, double kmax);

double plasmon_disp_inv(double w, double mu_e, double mu_h, double v_1,
    const system_data & sys);
double plasmon_disp_inv_ncb(double w, double mu_e, double mu_h, double v_1,
    const system_data & sys);

std::vector<double> plasmon_potcoef(const std::vector<double> & wkk,
    double mu_e, double mu_h, double v_1, const system_data & sys,
    double delta = 1e-2);

std::vector<double> plasmon_sysmat_full(const std::vector<uint32_t> & ids,
    const std::vector<double> & wkwk, double dw, double dk,
    const std::complex<double> & z, double mu_e, double mu_h, double v_1,
    const system_data & sys);

std::complex<double> plasmon_sysmat_det(
    const std::vector<std::complex<double>> & elems,
    const std::vector<uint32_t> & shape
);

std::vector<std::complex<double>> plasmon_sysmat_det_v(
    const std::vector<std::complex<double>> & z_vec,
    const std::vector<std::vector<double>> & wkwk,
    const std::vector<std::vector<uint32_t>> & ids,
    double dk, double dw,
    uint32_t N_k, uint32_t N_w,
    double mu_e, double mu_h, double v_1,
    const system_data & sys,
    double delta = 1e-2
);

std::complex<double> plasmon_sysmat_det_zero(
    const std::vector<std::vector<double>> & wkwk,
    const std::vector<std::vector<uint32_t>> & ids,
    double dk, double dw,
    uint32_t N_k, uint32_t N_w,
    double mu_e, double mu_h, double v_1,
    const system_data & sys,
    double delta = 1e-2
);

std::vector<double> plasmon_sysmat_det_lwl_v(
    const std::vector<double> & z_vec,
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids, double dk,
    uint32_t N_k, double ls, double v_1, const system_data & sys
);

std::vector<double> plasmon_sysmat_lwl_m(
    double z,
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids, double dk,
    double ls, double v_1, const system_data & sys
);

std::vector<std::complex<double>> plasmon_sysmat_lwl_eigvals(
    double z,
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids, double dk,
    uint32_t N_k, double ls, double v_1, const system_data & sys
);

double plasmon_sysmat_det_zero_lwl(
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids,
    double dk, uint32_t N_k, double ls, double v_1,
    const system_data & sys
);

std::vector<double> plasmon_real_potcoef_k(
    const std::vector<double> & x_vec,
    double mu_e, double mu_h, double v_1,
    const system_data & sys
);

