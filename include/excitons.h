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

double exciton_os(double Lx, double Ly, uint32_t nx, uint32_t ny);

double exciton_cm_se(
    double Lx, double Ly, uint32_t nx, uint32_t ny, const system_data& sys);

double exciton_cm_energy(uint32_t nx, uint32_t ny, const system_data& sys);

std::vector<double> exciton_d_vec(
    const std::vector<double>& energy,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys);

std::vector<double> exciton_d_nomb_vec(
    const std::vector<double>& energy,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys);

std::vector<double> exciton_lorentz_vec(
    const std::vector<double>& energy,
    double gamma,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys);

std::vector<double> exciton_lorentz_nomb_vec(
    const std::vector<double>& energy,
    double gamma,
    uint32_t nx,
    uint32_t ny,
    const system_data& sys);

std::vector<double> exciton_cont_vec(
    const std::vector<double>& energy_vec,
    double gamma_c,
    const system_data& sys);

double exciton_dd_var_E(
    double alpha,
    double rho_cm,
    double th_cm,
    double phi,
    double delta_z,
    const system_data& sys);

double exciton_dd_var_E_a0(
    double alpha,
    double a0,
    double rho_cm,
    double th_cm,
    double phi,
    double delta_z,
    const system_data& sys);
