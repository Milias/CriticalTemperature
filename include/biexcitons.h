#pragma once
#include "common.h"
#include "templates.h"
#include "wavefunction.h"

/*
 * param_alpha = |r'| / (Z * a0)
 * From page 123 of PhD 3.
 *
 * Computes the normalization of the wavefunction.
 */
double biexciton_norm_th(double param_alpha, double param_x);
double biexciton_norm_r(double param_alpha);
