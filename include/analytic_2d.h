/*
 * This file is the 2d version of "analytic.h",
 * therefore all functions implemented there should
 * have an equivalent here.
 */

#pragma once

#include "common.h"
#include "templates.h"
#include "wavefunction.h"
#include "analytic.h"

/*
 * Degree of ionization.
 *
 * Computes the degree of ionization for excitons
 * in the system. It is computed at high enough
 * temperature, so that the Maxwell-Boltzmann
 * distribution is a close enough approximation.
 *
 * The binding energy is computed from the screening
 * length given by "analytic_b_ex_E".
 */

double mb_2d_iod(double n, double lambda_s, const system_data & sys);
double mb_2d_iod_l(double n, double lambda_s, const system_data & sys);

/*** 2D Density ***/

/*
 * Same as in the 3D case.
 *
 * Note that here we take the binding energy of
 * the exciton as a parameter, not the scattering
 * length.
 */

double ideal_2d_n(double mu_i, double m_pi);
double analytic_2d_n_ex(double mu_t, double chi_ex, const system_data & sys);
double analytic_2d_n_sc(double mu_t, double chi_ex, const system_data & sys);

/*** Chemical potential ***/
/*
 * "analytic_2d_mu_ex" computes the excitonic
 * chemical potential given some excitonic
 * density and scattering length.
 */

double ideal_2d_mu(double n_id, const system_data & sys);
double ideal_2d_mu_h(double mu_e, const system_data & sys);
double ideal_2d_mu_v(double v, const system_data & sys);
double analytic_2d_mu_ex(double a, double n_ex, const system_data & sys);

/*** Scattering and screening lengths ***/

double ideal_2d_ls(double n_id, const system_data & sys);
double analytic_2d_a_ls(double ls, const system_data & sys);

/*** Solving for chemical potential ***/

std::vector<double> analytic_2d_mu(double n, const system_data & sys);

