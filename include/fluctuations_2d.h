/*
 * This file contains definitions for the functions used
 * in computing the full solution to the semiconductor (2D)
 * problem: determining the chemical potential as a function
 * of the carrier density n.
 *
 * Since it is an extension of "analytic_2d.h", some functions
 * used here will be defined in that file.
 */

#pragma once
#include "common.h"
#include "templates.h"
#include "analytic_2d.h"

std::complex<double> fluct_2d_I2(double z, double E, double mu_e, double mu_h, const system_data & sys);

