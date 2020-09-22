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
    const std::vector<double>& k0,
    const std::vector<double>& k,
    const system_data_v2& sys);
std::vector<std::complex<double>> topo_vert_2d_v(
    const std::vector<double>& k0,
    const std::vector<double>& k,
    const system_data_v2& sys);

std::vector<std::complex<double>> topo_cou_3d_v(
    const std::vector<double>& q1,
    const std::vector<double>& q2,
    const std::vector<double>& q,
    const system_data_v2& sys);
std::vector<std::complex<double>> topo_cou_2d_v(
    const std::vector<double>& q1,
    const std::vector<double>& q2,
    const std::vector<double>& q,
    const system_data_v2& sys);

std::vector<double> topo_det_p_cou_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_det_t_cou_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_det_t_eff_cou_cvcv_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

double topo_be_p_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd);
double topo_be_t_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd);

std::vector<double> topo_eff_cou_2d_mat(
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    uint8_t i,
    uint8_t j,
    const system_data_v2& sys);

double topo_be_t_eff_cou_cvcv(
    uint32_t N_k, const system_data_v2& sys, double be_bnd);
