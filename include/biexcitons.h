#pragma once
#include "common.h"
#include "templates.h"

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
result_s<1> biexciton_Jp_th(double param_alpha, double param_x1);
result_s<2> biexciton_Jp_r2(double param_alpha, double param_x1);
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
result_s<1> biexciton_Kp_r2(double param_alpha, double param_x2);
result_s<1> biexciton_Kp_r(double param_alpha);
