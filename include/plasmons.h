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

double plasmon_disp_th(
    const std::vector<double> wkk,
    double mu_e,
    double mu_h,
    const system_data& sys);

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
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_fmat_cx(
    const std::complex<double>& z,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_fmat_ht_cx(
    const std::complex<double>& z,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_det(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_det_ht(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_det_cx(
    const std::vector<std::complex<double>>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<std::complex<double>> plasmon_det_ht_cx(
    const std::vector<std::complex<double>>& z_vec,
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

double plasmon_det_zero(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

double plasmon_det_zero_ht(
    uint32_t N_k,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

double plasmon_det_zero_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

double plasmon_det_zero_ht_cx(
    uint32_t N_k,
    uint32_t N_w,
    double mu_e,
    double mu_h,
    const system_data& sys,
    double delta = 1e-12);

std::vector<double> plasmon_potcoef_lwl_v(
    const std::vector<std::vector<double>>& kk_vec,
    double ls,
    const system_data& sys);

std::vector<double> plasmon_mat_lwl(
    const std::vector<std::vector<double>>& vu_vec,
    const std::vector<std::vector<uint32_t>>& id_vec,
    uint32_t N_u,
    double du,
    double ls,
    double z,
    const system_data& sys);

double plasmon_det_zero_lwl(
    const std::vector<std::vector<double>>& vu_vec,
    const std::vector<std::vector<uint32_t>>& id_vec,
    uint32_t N_u,
    double du,
    double ls,
    const system_data& sys);

double plasmon_real_pot(
    double x, double mu_e, double mu_h, const system_data& sys);

std::vector<double> plasmon_real_pot_v(
    const std::vector<double>& x_vec,
    double mu_e,
    double mu_h,
    const system_data& sys);
