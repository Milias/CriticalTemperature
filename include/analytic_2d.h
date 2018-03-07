/*
 * This file is the 2d version of "analytic.h",
 * therefore all functions implemented there should
 * have an equivalent here.
 */

#pragma once

#include "common.h"
#include "templates.h"
#include "analytic.h"

/*
 * Exciton binding energy.
 *
 * The differential equation that fixes the ground-
 * state energy is different.
 */

double analytic_2d_b_ex_E(double lambda_s, const system_data & sys);

