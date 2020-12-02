#pragma once
#include "common.h"

std::vector<std::complex<double>> topo_ham_3d_v(
    double kx, double ky, double kz, const system_data_v2& sys);

std::vector<std::complex<double>> topo_orthU_3d_v(
    double kx, double ky, double kz, const system_data_v2& sys);
std::vector<std::complex<double>> topo_orthU_2d_v(
    double kx, double ky, const system_data_v2& sys);

std::vector<double> topo_eigenval_3d_v(
    double k, double kz, const system_data_v2& sys);
std::vector<double> topo_eigenval_2d_v(double k, const system_data_v2& sys);

std::vector<std::complex<double>> topo_vert_3d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys);
std::vector<std::complex<double>> topo_vert_2d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys);

std::vector<std::complex<double>> topo_cou_3d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys);
std::vector<std::complex<double>> topo_cou_2d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys);

double topo_disp_t_shift(const system_data_v2& sys);
double topo_disp_t_int(double Q, double k, const system_data_v2& sys);
double topo_disp_p_int(double Q, double k, const system_data_v2& sys);

std::vector<double> topo_det_p_cou_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_det_t_eff_cou_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_det_t_eff_cou_Q_vec(
    double Q,
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys);

std::vector<double> topo_det_p_cou_Q_vec(
    double Q,
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys);

double topo_be_p_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd);
double topo_be_p_cou_Q(
    double Q, uint32_t N_k, const system_data_v2& sys, double be_bnd);

std::vector<double> topo_cou_mat(uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_eff_cou_ij_mat(
    uint8_t i, uint8_t j, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_eff_cou_mat(uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_eff_cou_Q_ij_mat(
    double Q,
    uint8_t mat_i,
    uint8_t mat_j,
    uint32_t N_k,
    const system_data_v2& sys);

std::vector<double> topo_eff_cou_Q_mat(
    double Q, uint32_t N_k, const system_data_v2& sys);

double topo_be_t_eff_cou(
    uint32_t N_k, const system_data_v2& sys, double be_bnd);

double topo_be_t_eff_cou_Q(
    double Q, uint32_t N_k, const system_data_v2& sys, double be_bnd);

double topo_be_b_t_eff_cou_Q(
    double Q, uint32_t N_k, const system_data_v2& sys, double be_bnd);

std::vector<double> topo_p_cou_eig(
    double k_max, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_t_cou_eig(
    double k_max, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_t_eff_cou_eig(
    double Q, double k_max, uint32_t N_k, const system_data_v2& sys);

