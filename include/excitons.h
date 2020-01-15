#pragma once
#include "common.h"
#include "excitons_utils.h"
#include "templates.h"
#include "wavefunction_bexc.h"

std::vector<double> exciton_pot_cou_vec(
    const std::vector<double>& x_vec, const system_data& sys);
std::vector<double> exciton_wf_cou(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys);
double exciton_be_cou(const system_data& sys);

std::vector<double> exciton_pot_hn_vec(
    double size_d,
    double eps,
    const std::vector<double>& x_vec,
    const system_data& sys);

std::vector<double> exciton_wf_hn(
    double be_exc,
    double size_d,
    double eps,
    double r_max,
    uint32_t n_steps,
    const system_data& sys);

double exciton_be_hn(double size_d, double eps, const system_data& sys);

std::vector<double> exciton_pot_hnlr_vec(
    const std::vector<double>& x_vec, const system_data& sys);

std::vector<double> exciton_wf_hnlr(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys);

double exciton_be_hnlr(const system_data& sys);

std::vector<double> exciton_pot_ke_vec(
    const std::vector<double>& x_vec, const system_data& sys);

std::vector<double> exciton_wf_ke(
    double be_exc, double r_max, uint32_t n_steps, const system_data& sys);

double exciton_be_ke(const system_data& sys);
