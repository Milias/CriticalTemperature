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
 */

double ideal_2d_n(double mu_i, double m_pi);
double analytic_2d_n_ex(double mu_t, double a, const system_data & sys);
double analytic_2d_n_sc(double mu_t, double a, const system_data & sys);


/*** Chemical potential ***/

double ideal_2d_mu_h(double mu_e, const system_data & sys);

