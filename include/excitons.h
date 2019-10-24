#pragma once
#include "common.h"
#include "templates.h"
#include "wavefunction_bexc.h"

#include "excitons_utils.h"

std::vector<double> exciton_pot_cou_vec(
    const std::vector<double>& x_vec, const system_data& sys);
std::vector<double> exciton_wf_cou(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys);
double exciton_be_cou(const system_data& sys);

std::vector<double> exciton_pot_ke_vec(
    double size_d,
    double eps,
    const std::vector<double>& x_vec,
    const system_data& sys);

std::vector<double> exciton_wf_ke(
    double be_exc,
    double size_d,
    double eps,
    double r_max,
    uint32_t n_steps,
    const system_data& sys);

double exciton_be_ke(
    double size_d, double eps, const system_data& sys);

