/*
 * Here we define several helper structures
 * used in analytic.cpp.
 */

#pragma once

#include "common.h"

struct analytic_n_sc_s {
  double mu_t;
  double a;
};

struct ideal_mu_v_s {
  double v;
  const system_data & sys;
};

struct analytic_mu_s {
  double n;
  const system_data & sys;
};

struct analytic_b_ex_s {
  double lambda_s;
  const system_data & sys;
};
