#pragma once
#include "common.h"
#include "templates.h"
#include "wavefunction.h"

std::vector<std::complex<double>> plasmon_green_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_green_inv_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_green_ht_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_green_ht_inv_v(
    const std::vector<std::vector<double>> wk_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

double plasmon_kmax(double mu_e, double mu_h, const system_data& sys);
double plasmon_wmax(double mu_e, double mu_h, const system_data& sys);

double plasmon_disp(
    double k, double mu_e, double mu_h, const system_data& sys);
double plasmon_disp_ncb(
    double k, double mu_e, double mu_h, const system_data& sys, double kmax);

double plasmon_disp_inv(
    double w, double mu_e, double mu_h, const system_data& sys);
double plasmon_disp_inv_ncb(
    double w, double mu_e, double mu_h, const system_data& sys);

std::vector<std::complex<double>> plasmon_potcoef_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_potcoef_ht_cx_mat(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    const std::complex<double>& z,
    double delta = 1e-12);

double plasmon_det_zero(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta  = 1e-12);

double plasmon_det_zero_ht(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta  = 1e-12);

std::vector<double> plasmon_det_zero_v(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta = 1e-12);

std::vector<double> plasmon_det_zero_v1(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta = 1e-12);

std::vector<double> plasmon_det_zero_ht_v(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta = 1e-12);

std::vector<double> plasmon_det_zero_ht_v1(
    uint32_t N_k,
    const std::vector<double>& mu_vec,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta = 1e-12);

double plasmon_det_zero_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta  = 1e-12);

double plasmon_det_zero_ht_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double eb_min = std::numeric_limits<double>::quiet_NaN(),
    double delta  = 1e-12);

double plasmon_det_zero_lwl(uint32_t N_k, double ls, const system_data& sys);

double plasmon_rpot(
    double x, double mu_e, double mu_h, const system_data& sys);
double plasmon_rpot_ht(
    double x, double mu_e, double mu_h, const system_data& sys);

std::vector<double> plasmon_rpot_v(
    const std::vector<double>& x_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_rpot_ht_v(
    const std::vector<double>& x_vec,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_rpot_lwl_v(
    const std::vector<double>& x_vec,
    double ls,
    const system_data& sys,
    double delta = 1e-12);

double plasmon_exc_mu_zero(const system_data& sys);
double plasmon_exc_mu_val(double val, const system_data& sys);
double plasmon_exc_mu_lim_ht(
    uint32_t N_k,
    const system_data& sys,
    double val   = 0.0,
    double delta = 1e-12);

std::vector<double> plasmon_density_mu_ht_v(
    const std::vector<double>& u_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_density_ht_v(
    const std::vector<double>& n_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_density_exc_ht_v(
    const std::vector<double>& n_vec,
    uint32_t N_k,
    const system_data& sys,
    double delta = 1e-12);
