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

double fluct_2d_I2(double z, double E, double mu_e, double mu_h, const system_data & sys);
double fluct_2d_I2_dz(double z, double E, double mu_e, double mu_h, const system_data & sys);

std::complex<double> fluct_2d_I2_p(double z, double E, double chi_ex, double mu_e, double mu_h, const system_data & sys);

double fluct_2d_pp(double E, double chi_ex, double mu_e, double mu_h, const system_data & sys);

double fluct_2d_n_ex(double chi_ex, double mu_e, double mu_h, const system_data & sys);
double fluct_2d_n_sc(double chi_ex, double mu_e, double mu_h, const system_data & sys);
double fluct_2d_n_sc_v2(double chi_ex, double mu_e, double mu_h, const system_data & sys);

std::vector<double> fluct_2d_mu(double n, const system_data & sys);

