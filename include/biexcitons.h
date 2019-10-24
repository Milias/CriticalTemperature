#pragma once
#include "common.h"
#include "templates.h"
#include "wavefunction_bexc.h"

#include "biexcitons_utils.h"
/*
 * Solving the equation of state for the free charges +
 * excitons + biexcitons system.
 *
 * biexciton_eqst_c solves it for a constant value of the
 * biexciton binding energy.
 */
result_s<1> biexciton_eqst_u(
    double u,
    double be_exc,
    double be_biexc,
    double mu_e_lim,
    const system_data& sys);
result_s<2> biexciton_eqst_c(
    double n_gamma,
    double be_exc,
    double be_biexc,
    double u_init,
    const system_data& sys);
std::vector<double> biexciton_eqst_c_vec(
    const std::vector<double>& n_gamma_T_vec,
    double be_exc,
    double be_biexc,
    double u_init,
    const system_data& sys);

/*
 * param_alpha = |r'| / (Z * a0)
 * Computes the value of Delta: Eq. (14).
 */
result_s<1> biexciton_Delta_th(double param_alpha, double param_x);
result_s<1> biexciton_Delta_r(double r_BA, const system_data& sys);

/*
 * J integral: Eq. (17).
 */
result_s<2> biexciton_J_r(double r_BA, const system_data& sys);

/*
 * J' integral: Eq. (18).
 * biexciton_Jp_th is the angular part. param_alpha = r_BA / (Z * a0).
 * biexciton_Jp_r2 is the first radial part.
 * biexciton_Jp_r is the final radial part.
 */
result_s<2> biexciton_Jp_r2(double param_alpha, double param_x2);
result_s<1> biexciton_Jp_r(double r_BA, const system_data& sys);

/*
 * K integral: Eq. (20).
 */
result_s<1> biexciton_K_r(double r_BA, const system_data& sys);

/*
 * K' integral: Eq. (21).
 * param_th1 == param_th2 is a singular point.
 * param_x1 == param_x2 is another singular point.
 */
result_s<1> biexciton_Kp_th2(
    double param_alpha, double param_th1, double param_x1, double param_x2);
result_s<1> biexciton_Kp_th1(
    double param_alpha, double param_x1, double param_x2);

result_s<2> biexciton_Kp_r1(double param_alpha, double param_x2);
result_s<1> biexciton_Kp_r(double r_BA, const system_data& sys);

std::vector<result_s<1>> biexciton_Delta_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys);
std::vector<result_s<2>> biexciton_J_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys);
std::vector<result_s<1>> biexciton_Jp_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys);
std::vector<result_s<1>> biexciton_K_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys);
std::vector<result_s<1>> biexciton_Kp_r_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys);

std::vector<result_s<7>> biexciton_eff_pot_vec(
    const std::vector<double>& r_BA_vec, const system_data& sys);
std::vector<double> biexciton_eff_pot_interp_vec(
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    const std::vector<double>& x_interp_vec,
    const system_data& sys);

std::vector<double> biexciton_pot_r6_vec(
    double be_cou,
    const std::vector<double>& r_BA_vec,
    const system_data& sys);

std::vector<double> biexciton_wf_hl(
    double E,
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    uint32_t n_steps,
    const system_data& sys);

std::vector<double> biexciton_wf_r6(
    double E,
    double be_cou,
    double r_min,
    double r_max,
    uint32_t n_steps,
    const system_data& sys);

double biexciton_be_hl(
    double E_min,
    const std::vector<double>& x_vec,
    const std::vector<double>& pot_vec,
    const system_data& sys);

double biexciton_be_r6(
    double E_min, double be_cou, double r_min, const system_data& sys);
double biexciton_rmin_r6(
    double E_min, double be_cou, double be_biexc, const system_data& sys);

std::vector<double> biexciton_be_r6_vec(
    double E_min,
    double be_cou,
    const std::vector<double>& r_min_vec,
    const system_data& sys);

double biexciton_lj_c6(double be_cou, const system_data& sys);

std::vector<double> biexciton_pot_lj_vec(
    double param_c12,
    double be_cou,
    double r_max,
    uint32_t n_steps,
    const system_data& sys);
std::vector<double> biexciton_wf_lj(
    double be_biexc,
    double be_cou,
    double param_c12,
    double r_max,
    uint32_t n_steps,
    const system_data& sys);
double biexciton_be_lj(
    double param_c12, double be_cou, const system_data& sys);
std::vector<double> biexciton_be_lj_vec(
    const std::vector<double>& c12_vec, double be_cou, const system_data& sys);
double biexciton_c12_lj(
    double be_cou, double be_biexc, const system_data& sys);
