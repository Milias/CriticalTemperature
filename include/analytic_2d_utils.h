/*
 * Here we define several helper structures
 * used in analytic_2d.cpp.
 */

#pragma once

#include "common.h"

struct ideal_2d_mu_v_s {
  double v;
  const system_data & sys;
};

struct analytic_2d_n_sc_s {
  double mu_t;
  double chi_ex;

  const system_data & sys;
};

struct analytic_2d_mu_s {
  double n;
  const system_data & sys;
};

