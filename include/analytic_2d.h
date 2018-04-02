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

