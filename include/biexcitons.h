#pragma once
#include "common.h"
#include "templates.h"
#include "wavefunction_bexc.h"

#include "biexcitons_utils.h"

/*
 * param_alpha = |r'| / (Z * a0)
 * From page 123 of PhD 3.
 *
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
    double eb_cou,
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
    double eb_cou,
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
    double E_min, double eb_cou, double r_min, const system_data& sys);
double biexciton_rmin_r6(
    double E_min, double eb_cou, double eb_biexc, const system_data& sys);

std::vector<double> biexciton_be_r6_vec(
    double E_min,
    double eb_cou,
    const std::vector<double>& r_min_vec,
    const system_data& sys);
