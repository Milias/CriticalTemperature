#pragma once
#include "common.h"

std::vector<double> topo_det_p_cou_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

std::vector<double> topo_det_t_cou_vec(
    const std::vector<double>& z_vec, uint32_t N_k, const system_data_v2& sys);

double topo_be_p_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd);
double topo_be_t_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd);
