/*
 * Here we define several helper structures
 * used in fluctuations.cpp.
 */

#pragma once

#include "common.h"

struct fluct_T_i_s {
  constexpr static double c_f{- M_SQRT1_2 / 4.0};

  double z;
  double E;
  double mu_e;
  double mu_h;
  const system_data & sys;
};

struct fluct_T_z1_i_s {
  constexpr static double c_f{- M_SQRT1_2 / 4.0};

  double E;
  double mu_e;
  double mu_h;
  const system_data & sys;
};

